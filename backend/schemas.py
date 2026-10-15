from pydantic import BaseModel, Field


class ChurnPredictionRequest(BaseModel):
    tenure: int = Field(..., ge=0, description="Customer tenure in the organization")
    citytier: int = Field(..., ge=1, le=3, description="City tier (1, 2, or 3)")
    warehousetohome: int = Field(..., ge=0, description="Distance from warehouse to home")
    gender: str = Field(..., description="Customer gender (male/female)")
    hourspendonapp: int = Field(..., ge=0, description="Hours spent on the app")
    numberofdeviceregistered: int = Field(..., ge=0, description="Number of registered devices")
    satisfactionscore: int = Field(..., ge=1, le=5, description="Satisfaction score (1-5)")
    maritalstatus: str = Field(..., description="Marital status (single/married/divorced)")
    numberofaddress: int = Field(..., ge=0, description="Number of saved addresses")
    complain: int = Field(..., ge=0, le=1, description="Whether customer complained (0 or 1)")
    orderamounthikefromlastyear: int = Field(..., ge=0, description="Order amount hike % from last year")
    couponused: int = Field(..., ge=0, description="Number of coupons used")
    ordercount: int = Field(..., ge=0, description="Number of orders")
    daysincelastorder: int = Field(..., ge=0, description="Days since last order")
    cashbackamount: float = Field(..., ge=0, description="Average cashback amount")


class ChurnPredictionResponse(BaseModel):
    prediction: str
    churn_probability: float
    threshold: float
