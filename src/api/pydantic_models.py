from pydantic import BaseModel, Field
from typing import Optional

class CustomerData(BaseModel):
    # Define all model input features here, example:
    Value_sum: float
    Value_mean: float
    Value_count: float
    Value_std: float
    Amount_mean: float
    Amount_std: float
    transaction_hour: Optional[int] = Field(None, description="Transaction hour")
    transaction_day: Optional[int] = Field(None, description="Transaction day")
    transaction_month: Optional[int] = Field(None, description="Transaction month")
    # Add any other features you engineered, with correct types
