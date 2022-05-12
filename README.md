# Image Recognition Final Project
# Microservice Implementation with FastAPI

<!-- ABOUT THE PROJECT -->
## Contributors
- **Chukwudi Ikem**
- **Nolan O’Donnell**

### Built With

* [Python](https://www.python.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [Foreman](https://pypi.org/project/foreman/)
* [Uvicorn](https://www.uvicorn.org/)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Steps

1. Clone the repo
   ```sh
   git clone https://github.com/Super-Rogatory/thepercentfortyfive
   ```
2. Initialize Project! (you may need to change permissions on your local machine)
   ```sh
   ./init.sh
   ```    
3. Run the model file to train our model (you can tweak it to increase accuracy)
   ```sh
   cd /ml
   python3 model.py
   ```
4. Test the model file to check accuracy rating (default at 45%)
   ```sh
   python3 test.py
   ```   
5. Start up FastAPI Server
   ```sh
   foreman start server
   ```   
6. Travel to http://127.0.0.1:5000

