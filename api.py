import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from modules.inference import CategorizerInference
from modules.tools import clean_text, Formatter
from pydantic import BaseModel

model_path = os.getenv("MODEL_PATH")
if model_path is None:
    model_path = "models/IRIS-BERT-base-Categorizer"

print("MODEL_PATH:", model_path)
print("Loading models...")
models = CategorizerInference(model_path)
print("Models loaded")

formatter = Formatter("formats/default.txt")

app = FastAPI()

@app.get("/") 
async def read_root():
    return {"Hello": "World"}

@app.post("/tickets/Categorize")
async def categorize(ticket_text: str):
    """
    Calculates the categories and properties of a ticket text
    Params are in JSON format {"ticket_text": "text"}
    :param: ticket_text: str: The text of the ticket. Text formatting example: Partner: {partner} | Name: {contact} | Subject: {subject} | Message: {description} 
    :return: {}
    """
    if ticket_text is None:
        return JSONResponse(content={"error": "ticket_text is required"}, status_code=400)
    
    json_output = models.infer(ticket_text)

    return JSONResponse(content=json_output, status_code=200)

class Ticket(BaseModel):
    partner: str
    contact: str
    subject: str
    description: str

@app.post("/tickets/CategorizeJson", response_model=Ticket)
async def categorize_json(request: Request):
    """
    Calculates the categories and properties of a ticket text
    """
    ticket = await request.json()
    
    # check the necessary fields
    if "partner" not in ticket:
        return JSONResponse(content={"error": "partner is required"}, status_code=400)
    if "contact" not in ticket:
        return JSONResponse(content={"error": "contact is required"}, status_code=400)
    if "subject" not in ticket:
        return JSONResponse(content={"error": "subject is required"}, status_code=400)
    if "description" not in ticket:
        return JSONResponse(content={"error": "description is required"}, status_code=400)
    
    ticket["description"] = clean_text(ticket["description"])
    ticket_text = formatter(ticket)

    json_output = models.infer(ticket_text, ticket["partner"])

    return JSONResponse(content=json_output, status_code=200)

