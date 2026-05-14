from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, JSON
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Personaje(Base):
    __tablename__ = "personajes"

    id = Column(Integer, primary_key=True)
    nombre = Column(String(255), nullable=False)
    raza_id = Column(Integer, ForeignKey("razas.id"))
    lugar_origen_id = Column(Integer, ForeignKey("lugares.id"))
    faccion_id = Column(Integer, ForeignKey("facciones.id"))

    raza = relationship("Raza", back_populates="personajes")
    lugar_origen = relationship("Lugar", back_populates="personajes_origen")
    faccion = relationship("Faccion", back_populates="personajes")
    artefactos = relationship("ArtefactoMagico", back_populates="portador")


class Raza(Base):
    __tablename__ = "razas"

    id = Column(Integer, primary_key=True)
    nombre = Column(String(100), nullable=False, unique=True)
    descripcion = Column(Text)

    personajes = relationship("Personaje", back_populates="raza")


class Lugar(Base):
    __tablename__ = "lugares"

    id = Column(Integer, primary_key=True)
    nombre = Column(String(255), nullable=False)
    tipo = Column(String(50))

    personajes_origen = relationship("Personaje", back_populates="lugar_origen")


class Faccion(Base):
    __tablename__ = "facciones"

    id = Column(Integer, primary_key=True)
    nombre = Column(String(255), nullable=False, unique=True)
    tipo = Column(String(50))

    personajes = relationship("Personaje", back_populates="faccion")


class ArtefactoMagico(Base):
    __tablename__ = "artefactos_magicos"

    id = Column(Integer, primary_key=True)
    nombre = Column(String(255), nullable=False)
    portador_id = Column(Integer, ForeignKey("personajes.id"))

    portador = relationship("Personaje", back_populates="artefactos")


class ExtractionLog(Base):
    """Registro de cada llamada al endpoint /extract."""

    __tablename__ = "extraction_logs"

    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(Text, nullable=False)
    entity_count = Column(Integer, nullable=False, default=0)
    entities_found = Column(
        JSON, nullable=True
    )  # snapshot [{text, label, start, end}, ...]
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
