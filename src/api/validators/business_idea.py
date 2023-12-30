from pydantic import BaseModel


class CountryName(BaseModel):
    country_name: str


class Business_Analysis(BaseModel):
    country_name: str
    current_year: int
    business_idea: str
    business_analysis: str
    financial_data: str
