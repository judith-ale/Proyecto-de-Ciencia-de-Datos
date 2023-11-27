import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security, UploadFile
import tensorflow as tf
from fastapi.security.api_key import APIKeyQuery, APIKey
from starlette.status import HTTP_403_FORBIDDEN
import mlflow

app = FastAPI()


# Simulación de seguridad
API_KEY = "1234567asdfgh"
API_KEY_NAME = "access_token"

api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key_query: str = Security(api_key_query)):

    if api_key_query == API_KEY:
        return api_key_query
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )


dictLetras = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'l': 9,
    'm': 10,
    'n': 11,
    'o': 12,
    'p': 13,
    'r': 14,
    's': 15,
    't': 16,
    'u': 17,
    'v': 18,
    'w': 19,
    'y': 20,
    'z': 21
}

invDictLetras = {value: key for key, value in dictLetras.items()}

mlflow.set_tracking_uri('https://dagshub.com/judith-ale/Proyecto-de-Ciencia-de-Datos.mlflow')
logged_model = 'runs:/2434901c0d104310ab967964789740aa/InceptionV3_model'

# Cargar modelo
loaded_model = mlflow.tensorflow.pyfunc.load_model(logged_model)


@app.get("/api/v0/jpeg/classify")
async def predict_letra(
        file: UploadFile,
        api_key: APIKey = Depends(get_api_key)
):
    # Tamaño del archivo
    file.file.seek(0, 2)
    file_size = file.file.tell()

    await file.seek(0)

    # Más de 2 MB
    if file_size > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    # Contenido distinto a imagen jpeg
    if file.content_type != "image/jpeg":
        raise HTTPException(status_code=400, detail="Invalid file type")

    contents = await file.read()

    img = tf.image.decode_jpeg(contents, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float16)
    img = tf.image.resize(img, (200, 200))
    img = img.numpy()
    img = img.reshape([1, *img.shape])

    # Predicción
    prediction = loaded_model.predict(img)

    return {"Letra": invDictLetras[prediction.argmax()]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=False)
