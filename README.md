# Bike Rental Chatbot

An intelligent chatbot system for a bike rental business that helps customers at different stages of their rental journey.

## Features

- **Pre-rental Assistance**
  - Vehicle recommendations based on criteria
  - Price range filtering
  - Mileage and feature-based suggestions

- **During Rental Support**
  - Feature inquiries
  - Basic troubleshooting guidance
  - Emergency contact information

- **Post-rental Services**
  - Feedback collection
  - Service rating system

## Technical Stack

- **LLM**: HuggingFace Transformers
- **Framework**: LangChain
- **Database**: PostgreSQL
- **API**: FastAPI
- **Vector Store**: SQLAlchemy

## Project Structure

```
bike_rental_chatbot/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   ├── config.py
│   │   └── database.py
│   ├── models/
│   │   ├── vehicle.py
│   │   └── chat.py
│   ├── services/
│   │   ├── chatbot.py
│   │   └── vehicle_service.py
│   └── utils/
│       └── helpers.py
├── data/
│   └── vehicle_data.sql
├── tests/
│   └── test_chatbot.py
├── .env
├── requirements.txt
└── main.py
```

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in the required environment variables

4. Initialize the database:
   ```bash
   python scripts/init_db.py
   ```

5. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

## Database Schema

The database includes the following main tables:

- **vehicles**
  - vehicle_id (PK)
  - vehicle_type (4-wheeler/2-wheeler)
  - transmission_type (geared/non-geared)
  - last_service_date
  - terrain_suitability
  - mileage
  - vehicle_rating
  - price_per_day
  - availability_status
  - model_name
  - brand
  - year_of_manufacture

- **rentals**
  - rental_id (PK)
  - vehicle_id (FK)
  - customer_id (FK)
  - start_date
  - end_date
  - status
  - total_cost

- **feedback**
  - feedback_id (PK)
  - rental_id (FK)
  - rating
  - comments
  - created_at

## API Endpoints

- `POST /api/chat`: Chat with the bot
- `GET /api/vehicles`: List available vehicles
- `GET /api/vehicles/{id}`: Get vehicle details
- `POST /api/feedback`: Submit feedback

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

