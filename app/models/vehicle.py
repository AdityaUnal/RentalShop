from sqlalchemy import Column, Integer, String, Float, Date, Boolean, Enum
from app.core.database import Base
import enum

class VehicleType(str, enum.Enum):
    TWO_WHEELER = "2-wheeler"
    FOUR_WHEELER = "4-wheeler"

class TransmissionType(str, enum.Enum):
    GEARED = "geared"
    NON_GEARED = "non-geared"

class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_type = Column(Enum(VehicleType))
    transmission_type = Column(Enum(TransmissionType))
    model_name = Column(String)
    brand = Column(String)
    year_of_manufacture = Column(Integer)
    last_service_date = Column(Date)
    terrain_suitability = Column(String)  # Comma-separated list of terrains
    mileage = Column(Float)  # km/l
    vehicle_rating = Column(Float)  # 1-5 rating
    price_per_day = Column(Float)
    availability_status = Column(Boolean, default=True)
    features = Column(String)  # JSON string of features
    maintenance_history = Column(String)  # JSON string of maintenance records
    insurance_info = Column(String)  # JSON string of insurance details
    registration_number = Column(String, unique=True)
    color = Column(String)
    fuel_type = Column(String)
    engine_capacity = Column(Float)  # in cc
    seating_capacity = Column(Integer)
    created_at = Column(Date)
    updated_at = Column(Date) 