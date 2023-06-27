# This is a basic chat bot that reads in PDF files
- LLAMA 
- LangChain
- Streamlit 
- OpenAPI 

The chat bot is fed a pdf or any type of readable text file from the dockerfile which is queried by the container. 
- The API key is a enviroment secret placed on the kubernetes cluster 
- Underdevelopment to add vector DB and change the static files to a array of urls with the loader function from LLAM. 