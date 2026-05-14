from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship, declarative_base

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
