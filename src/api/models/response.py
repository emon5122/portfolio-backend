from typing import List

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql.expression import text

from core.database import Base


# tag wise response
class Tags(Base):
    __tablename__ = "tags"
    id: Mapped[int] = mapped_column(primary_key=True)
    tag = Column(String, nullable=False)
    responses: Mapped[List["Responses"]] = relationship()


class Responses(Base):
    __tablename__ = "responses"
    id: Mapped[int] = mapped_column(primary_key=True)
    response = Column(String, nullable=False)
    tag_id: Mapped[int] = mapped_column(ForeignKey("tags.id"))
