# RentalShop
- Designed a chatbot for a Vehicle Rental Service.
- The UI is using streamlit.
- The backend has been created using langraph, langchain, huggingface tokens.

## Getting Started
- Make sure you are in a Virtual Environment([Reference]([url](https://www.w3schools.com/python/python_virtualenv.asp))).
```shell
git clone https://github.com/AdityaUnal/RentalShop.git
pip install -r requirements.txt
touch .env
# Fill the .env file, refer below to see how it should look like
streamlit run app.py
```
- .env file : 
```text
HUGGINGFACE_API_KEY=hf_xxxxxxx
TAVILY_API_KEY=tvly-dev-xxxxx
```
- [TAVILY_API_KEY]([url](https://docs.tavily.com/documentation/quickstart))
- [HUGGINGFACE_API_KEY]([url](https://www.geeksforgeeks.org/how-to-access-huggingface-api-key/))
- Make sure that you have given read writes to your huggingface api key

## Sample Screenshots

![image](https://github.com/user-attachments/assets/5df213c9-2292-4f18-9000-9963dc32f055)
![image](https://github.com/user-attachments/assets/222d57fa-cea0-4c72-a17f-7baad73b629a)

## Tree
```Tree
.
├── chroma_db
├── db
├── .env
├── .gitignore
├── .python-version
├── README.md
├── Travel Partner.pdf
├── app.py
├── chabot.ipynb
├── pyproject.toml
├── requirements.txt
└── uv.lock
```
