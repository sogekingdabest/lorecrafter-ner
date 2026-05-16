import json
import os
import asyncio
from pathlib import Path
import yaml
import time
import re
from groq import AsyncGroq

async_client = None

def get_async_client(llm_config):
    global async_client
    if async_client is None:
        async_client = AsyncGroq()
    return async_client

def log_interaction(prompt, response, status="SUCCESS", error=None):
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "llm_requests.log"
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] STATUS: {status}\n")
        f.write(f"--- REQUEST ---\n{prompt}\n")
        if response:
            f.write(f"--- RESPONSE ---\n{response}\n")
        if error:
            f.write(f"--- ERROR ---\n{error}\n")
        f.write("="*80 + "\n\n")

SYSTEM_PROMPT = """You are an expert NER (Named Entity Recognition) annotator specializing in fantasy literature.

LABELS:
- PERSONAJE: Character names (individuals, gods, beings with proper names)
- FACCION: Groups, organizations, guilds, orders, armies, councils, factions
- LUGAR: Cities, kingdoms, regions, buildings, geographic locations
- ARTEFACTO_MAGICO: Magical objects, enchanted weapons, artifacts with supernatural properties
- RAZA: Fantasy races, species, creature types (Elf, Dwarf, Orc, etc.)

TASK:
Annotate ALL named entities in the provided text.

RULES:
1. Annotate ONLY proper nouns and specific named entities
2. Do NOT annotate generic terms like "sword", "city", "king" unless part of a proper name
3. Include full compound names ("Aragorn son of Arathorn" as one PERSONAJE)
4. For race names used as proper nouns (e.g., "the Elves"), annotate as RAZA
5. For faction names that include locations, annotate the full faction name

OUTPUT FORMAT (JSON only, no markdown):
{
  "entities": [
    {"entity": "exact string from text", "label": "LABEL"},
    ...
  ]
}

The "entity" value MUST be an exact substring that appears in the provided text. Do NOT use numerical offsets.
Return ONLY valid JSON. Do not include explanations."""

def load_config(config_path="configs/llm_generation.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["llm"], config["preannotation"]

async def annotate_text(text, llm_config):
    full_prompt = f"{SYSTEM_PROMPT}\n\n---\n\nAnnotate entities in this text:\n\n{text}"
    
    max_retries = 8
    retry_delay = 7

    for attempt in range(max_retries):
        try:
            cli = get_async_client(llm_config)
            response = await cli.chat.completions.create(
                model=llm_config["model"],
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.1,
                top_p=0.9,
            )

            content = response.choices[0].message.content.strip()
            log_interaction(full_prompt, content, status="SUCCESS")

            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)

            if "entities" not in parsed:
                return None

            exact_entities = []
            for ent_dict in parsed["entities"]:
                if isinstance(ent_dict, list):
                    continue
                
                ent_text = ent_dict.get("entity", ent_dict.get("text"))
                label = ent_dict.get("label")
                
                if not ent_text or not label:
                    continue
                
                if label not in ["PERSONAJE", "FACCION", "LUGAR", "ARTEFACTO_MAGICO", "RAZA"]:
                    continue

                pattern = re.escape(str(ent_text))
                for match in re.finditer(pattern, text):
                    start, end = match.span()
                    if not any(e[0] == start and e[1] == end for e in exact_entities):
                        exact_entities.append([start, end, label])

            exact_entities.sort(key=lambda x: x[0])
            
            filtered_entities = []
            for e in exact_entities:
                overlap = False
                for fe in filtered_entities:
                    if (e[0] < fe[1] and e[1] > fe[0]):
                        overlap = True
                        break
                if not overlap:
                    filtered_entities.append(e)

            return filtered_entities

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                log_interaction(full_prompt, None, status="RATE_LIMIT", error=error_str)
                print(f"  [Rate Limit 429] Esperando {retry_delay}s antes de reintentar... (Intento {attempt+1}/{max_retries})")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 60)
            else:
                log_interaction(full_prompt, None, status="ERROR", error=error_str)
                print(f"  Error annotating text: {e}")
                return None

    log_interaction(full_prompt, None, status="FAILED_MAX_RETRIES", error="Rate limit retries exhausted.")
    print(f"  Fallo tras {max_retries} reintentos por Rate Limit.")
    return None

def load_scraped_texts(input_path="data/raw/scraped_texts.txt"):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("---SEPARATOR---")
    texts = [block.strip() for block in blocks if block.strip()]

    chunks = []
    for text in texts:
        sentences = text.replace("\n", " ").split(". ")
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) < 500:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip() and len(current_chunk.strip()) > 50:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk.strip() and len(current_chunk.strip()) > 50:
            chunks.append(current_chunk.strip())

    print(f"Loaded {len(texts)} text blocks, split into {len(chunks)} chunks")
    return chunks

async def worker(text, llm_config, semaphore):
    async with semaphore:
        return text, await annotate_text(text, llm_config)

async def preannotate_dataset(
    input_path="data/raw/scraped_texts.txt",
    output_path="data/annotations/llm_preannotated.json",
    batch_size=20,
):
    llm_config, preann_config = load_config()

    output_path = preann_config.get("output_path", output_path)
    batch_size = preann_config.get("batch_size", batch_size)
    max_concurrent = preann_config.get("max_concurrent", 2)

    texts = load_scraped_texts(input_path)

    print(f"Pre-annotating {len(texts)} text chunks with {llm_config['model']}...")
    print(f"Batch size: {batch_size}, Max concurrent requests: {max_concurrent}")

    all_examples = []
    output_file = Path(output_path)
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_examples = json.load(f)
            print(f"Resuming... Loaded {len(all_examples)} existing examples from {output_path}")
        except Exception as e:
            print(f"Warning: Could not read existing file (starting from scratch): {e}")

    start_idx = len(all_examples)
    if start_idx >= len(texts):
        print(f"Already reached the target count of {len(texts)}. Exiting.")
        return all_examples
        
    remaining = len(texts) - start_idx
    print(f"Remaining examples to generate: {remaining}")

    errors = 0
    semaphore = asyncio.Semaphore(max_concurrent)

    for batch_start in range(start_idx, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        print(f"\nBatch ({batch_start}-{batch_end}) / Total {len(texts)}")
        
        tasks = []
        for i in range(batch_start, batch_end):
            tasks.append(worker(texts[i], llm_config, semaphore))
            
        results = await asyncio.gather(*tasks)
        
        for text, entities in results:
            if entities is not None:
                all_examples.append({"text": text, "entities": entities})
            else:
                errors += 1
                
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_examples, f, ensure_ascii=False, indent=2)
            
        print(f"  Progress saved: {len(all_examples)}/{len(texts)} annotated ({errors} errors so far)")
        
        await asyncio.sleep(1)

    print(f"\nPre-annotation complete:")
    print(f"  Annotated: {len(all_examples)}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_path}")

    label_counts = {}
    for ex in all_examples:
        for _, _, label in ex["entities"]:
            label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nEntity distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    return all_examples

if __name__ == "__main__":
    asyncio.run(preannotate_dataset())
