import json
import asyncio
from pathlib import Path
import yaml
import time
import random
import re
from groq import AsyncGroq


async_client = None


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
        f.write("=" * 80 + "\n\n")


def get_async_client(llm_config):
    global async_client
    if async_client is None:
        async_client = AsyncGroq()
    return async_client


SYSTEM_PROMPT = (
    "You are an expert fantasy writer and NER annotator. "
    "Your task is to generate realistic fantasy narrative text "
    "and annotate all named entities.\n\n"
    "LABELS:\n"
    "- PERSONAJE: Character names (individuals, gods, beings with names)\n"
    "- FACCION: Groups, organizations, guilds, orders, armies, councils\n"
    "- LUGAR: Cities, kingdoms, regions, buildings, geographic features\n"
    "- ARTEFACTO_MAGICO: Magical objects, enchanted weapons, artifacts with power\n"
    "- RAZA: Fantasy races, species, creature types\n\n"
    "RULES:\n"
    "1. Write 2-5 sentence narratives in a literary fantasy style\n"
    "2. Include AMBIGUITIES and complexities:\n"
    "   - Pronouns referring to previously mentioned entities\n"
    '   - Generic terms vs specific names ("the sword" vs "Anduril")\n'
    '   - Compound names ("Aragorn son of Arathorn")\n'
    "   - Entities that could be confused "
    "(Elf names that sound like city names)\n"
    '   - Context-dependent labels ("the north" as direction vs faction)\n'
    '   - Partial mentions ("he drew his blade" without naming it)\n'
    "3. Use varied sentence structures: complex, compound, "
    "with subordinate clauses\n"
    "4. Include dialogue occasionally\n"
    "5. Mix English and Spanish fantasy prose styles\n"
    "6. Do NOT use template-like patterns. "
    "Each narrative should feel unique.\n\n"
    "OUTPUT FORMAT (JSON only, no markdown):\n"
    '{"text": "full narrative text", "entities": '
    '[{"entity": "exact string", "label": "LABEL"}, ...]}\n\n'
    'The "entity" value MUST be an exact substring that appears '
    'in the generated "text". Do NOT use numerical offsets.'
)

AMBIGUITY_PROMPTS = [
    (
        "Include a scene where a character's race is mentioned ambiguously "
        "(e.g., 'the elf' could refer to the person or the race)."
    ),
    (
        "Write about a magical artifact that is only described, not named, "
        "alongside one that is named."
    ),
    (
        "Include a faction that shares a name with a location "
        "(e.g., 'the Riders of Rohan' vs 'Rohan')."
    ),
    "Write a passage with compound character names and family lineages.",
    (
        "Include dialogue where characters refer to each other "
        "by titles rather than names."
    ),
    "Write about a place that is also the name of a people or faction.",
    (
        "Include a scene with multiple entities of the same type "
        "that could be confused."
    ),
    (
        "Write a passage where the same word could be interpreted as "
        "different entity types depending on context."
    ),
    (
        "Include a description of a magical object without explicitly "
        "stating it's magical."
    ),
    ("Write about a journey through multiple locations " "with faction encounters."),
]


def load_config(config_path="configs/llm_generation.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["llm"], config["generation"]


async def generate_single_example(llm_config, ambiguity_prompt=None):
    prompt = "Generate a fantasy narrative scene. "
    if ambiguity_prompt:
        prompt += f"Specific requirement: {ambiguity_prompt} "
    prompt += "Remember to return ONLY valid JSON with exact character offsets."

    full_prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{prompt}"

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
                temperature=llm_config.get("temperature", 0.7),
                top_p=llm_config.get("top_p", 0.9),
            )

            content = response.choices[0].message.content.strip()
            log_interaction(full_prompt, content, status="SUCCESS")

            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)

            if "text" not in parsed or "entities" not in parsed:
                return None

            generated_text = parsed["text"]
            exact_entities = []

            for ent_dict in parsed["entities"]:
                if isinstance(ent_dict, list):
                    continue  # Por si el modelo se equivoca y manda una lista vieja

                ent_text = ent_dict.get("entity")
                label = ent_dict.get("label")

                if not ent_text or not label:
                    continue

                pattern = re.escape(str(ent_text))
                for match in re.finditer(pattern, generated_text):
                    start, end = match.span()
                    if not any(e[0] == start and e[1] == end for e in exact_entities):
                        exact_entities.append([start, end, label])

            exact_entities.sort(key=lambda x: x[0])

            filtered_entities = []
            for e in exact_entities:
                overlap = False
                for fe in filtered_entities:
                    if e[0] < fe[1] and e[1] > fe[0]:
                        overlap = True
                        break
                if not overlap:
                    filtered_entities.append(e)

            parsed["entities"] = filtered_entities
            return parsed

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                log_interaction(full_prompt, None, status="RATE_LIMIT", error=error_str)
                # Sacar el número de segundos si Groq lo provee en el mensaje, o usar el default
                print(
                    f"  [Rate Limit 429] Esperando {retry_delay}s "
                    f"antes de reintentar... "
                    f"(Intento {attempt+1}/{max_retries})"
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(
                    retry_delay * 1.5, 60
                )  # Backoff exponencial tapado a 60s
            else:
                log_interaction(full_prompt, None, status="ERROR", error=error_str)
                print(f"  Error generating example: {e}")
                return None

    log_interaction(
        full_prompt,
        None,
        status="FAILED_MAX_RETRIES",
        error="Rate limit retries exhausted.",
    )
    print(f"  Fallo tras {max_retries} reintentos por Rate Limit.")
    return None


async def worker(llm_config, ambiguity_prompt, semaphore):
    async with semaphore:
        return await generate_single_example(llm_config, ambiguity_prompt)


async def generate_synthetic_dataset(
    count=3500, batch_size=50, output_path="data/annotations/llm_synthetic.json"
):
    llm_config, gen_config = load_config()

    count = gen_config.get("synthetic_count", count)
    batch_size = gen_config.get("batch_size", batch_size)
    output_path = gen_config.get("output_path", output_path)

    # max_concurrent defines how many parallel calls to Ollama are allowed
    max_concurrent = gen_config.get("max_concurrent", 5)

    print(f"Generating {count} synthetic examples with {llm_config['model']}...")
    print(f"Batch size: {batch_size}, Max concurrent requests: {max_concurrent}")

    all_examples = []

    # 1. Resumability: Load existing examples if they exist
    output_file = Path(output_path)
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_examples = json.load(f)
            print(
                f"Resuming... Loaded {len(all_examples)} existing examples from {output_path}"
            )
        except Exception as e:
            print(f"Warning: Could not read existing file (starting from scratch): {e}")

    start_idx = len(all_examples)
    if start_idx >= count:
        print(f"Already reached the target count of {count}. Exiting.")
        return all_examples

    remaining = count - start_idx
    print(f"Remaining examples to generate: {remaining}")

    errors = 0
    ambiguity_idx = start_idx
    semaphore = asyncio.Semaphore(max_concurrent)

    # 2. Concurrency: Process in batches with asyncio
    for batch_start in range(start_idx, count, batch_size):
        batch_end = min(batch_start + batch_size, count)

        print(f"\nBatch ({batch_start}-{batch_end}) / Total {count}")

        tasks = []
        for i in range(batch_start, batch_end):
            if i % 10 == 0:
                ambiguity_prompt = AMBIGUITY_PROMPTS[
                    ambiguity_idx % len(AMBIGUITY_PROMPTS)
                ]
                ambiguity_idx += 1
            else:
                ambiguity_prompt = None

            tasks.append(worker(llm_config, ambiguity_prompt, semaphore))

        # Execute batch concurrently
        results = await asyncio.gather(*tasks)

        for res in results:
            if res:
                all_examples.append(res)
            else:
                errors += 1

        # Save progress after every batch
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_examples, f, ensure_ascii=False, indent=2)

        print(
            f"  Progress saved: {len(all_examples)}/{count} generated ({errors} errors so far)"
        )

        await asyncio.sleep(1)

    print("\nGeneration complete:")
    print(f"  Total examples: {len(all_examples)}")
    print(f"  Errors encountered: {errors}")
    print(f"  Output saved to: {output_path}")

    label_counts = {}
    for ex in all_examples:
        for _, _, label in ex["entities"]:
            label_counts[label] = label_counts.get(label, 0) + 1

    print("\nEntity distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    return all_examples


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(generate_synthetic_dataset())
