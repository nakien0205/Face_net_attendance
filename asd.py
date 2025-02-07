# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir, join

from PIL import Image
from facenet_pytorch import InceptionResnetV1
# extract a single face from a given photograph
from facenet_pytorch import MTCNN
from numpy import asarray, expand_dims, savez_compressed, load


# Replace the `load_model` function to initialize InceptionResnetV1 from facenet_pytorch
def load_model(path: str = None):
    try:
        # Initialize the InceptionResnetV1 model pretrained on VGGFace2
        model = InceptionResnetV1(pretrained='vggface2').eval()
        print("Model initialized successfully (InceptionResnetV1).")
        return model
    except Exception as e:
        # Catch and report any errors during model initialization
        raise RuntimeError(f"Error initializing model: {str(e)}")





# Extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    try:
        # Load image from file
        image = Image.open(filename)
        # Convert to RGB, if needed
        image = image.convert('RGB')
        # Convert to array
        pixels = asarray(image)
        # Create the detector, using default weights
        detector = MTCNN()
        # Detect faces in the image
        detection_results = detector.detect(pixels)
        # Handle case if there are no detection results
        if detection_results is None:
            raise ValueError(f"No face detected in image: {filename}")

        # Unpack detection results
        boxes, _ = detection_results[:2]  # Ensure only boxes are used
        # Ensure at least one face is detected
        if boxes is None or len(boxes) == 0:
            raise ValueError(f"No face detected in image: {filename}")

        # Extract the bounding box from the first detected face
        x1, y1, x2, y2 = boxes[0]
        x1, y1 = int(max(x1, 0)), int(max(y1, 0))  # Ensure coordinates are non-negative
        x2, y2 = int(x2), int(y2)
        # Extract the face
        face = pixels[y1:y2, x1:x2]
        # Resize pixels to the required model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array

    except (IndexError, ValueError) as e:
        print(f"Error processing file {filename}: {str(e)}")
        return None


# Load images and extract faces for all images in a directory
def load_faces(directory):
    faces = []
    # Enumerate files
    for filename in listdir(directory):
        # Path
        path = join(directory, filename)
        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        # Get face
        face = extract_face(path)
        if face is not None:  # Only append valid faces
            faces.append(face)
    return faces


# Load a dataset that contains one subdirectory for each class (each containing images)
def load_dataset(directory):
    X, y = [], []
    # Enumerate folders, one per class
    for subdir in listdir(directory):
        # Path
        path = join(directory, subdir)
        # Skip any files that might be in the directory
        if not isdir(path):
            continue
        # Load all faces in the subdirectory
        faces = load_faces(path)
        if len(faces) == 0:
            print(f"Warning: No faces found in class directory {subdir}")
            continue
        # Create labels
        labels = [subdir] * len(faces)
        # Summarize progress
        print(f">Loaded {len(faces)} examples for class: {subdir}")
        # Store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


# Load train dataset
trainX, trainy = load_dataset('data/train/')
# Load test dataset
testX, testy = load_dataset('data/val/')
# Save arrays to one file in compressed format
savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)

# Constants
DATASET_PATH = '5-celebrity-faces-dataset.npz'
MODEL_PATH = 'facenet_keras.h5'


# Generate face embedding for one face
def generate_face_embedding(model, face_image: asarray) -> asarray:
    # Scale and standardize pixel values
    face_image = face_image.astype('float32')
    mean, std = face_image.mean(), face_image.std()
    face_image = (face_image - mean) / std
    # Transform the face into a single sample for prediction
    sample = expand_dims(face_image, axis=0)
    # Make prediction to get embedding
    embedding = model.predict(sample)  # Assuming `model.predict` exists for the given model
    return embedding[0]


# Convert a set of faces to embeddings
def convert_faces_to_embeddings(model, faces: asarray) -> asarray:
    embeddings = [generate_face_embedding(model, face_image) for face_image in faces]
    return asarray(embeddings)


# Load the dataset
data = load(DATASET_PATH)
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded dataset: ', trainX.shape, trainy.shape, testX.shape, testy.shape)


# Load the custom model
def load_model(path: str):
    try:
        # Load the model using TensorFlow/Keras
        model = keras_load_model(path)
        print("Model loaded successfully from:", path)
        return model
    except Exception as e:
        # Catch and report any errors during model loading
        raise RuntimeError(f"Error loading model from {path}: {str(e)}")


# Load the model
model = load_model(MODEL_PATH)
print('Loaded Model')

# Generate embeddings for train and test datasets
train_embeddings = convert_faces_to_embeddings(model, trainX)
test_embeddings = convert_faces_to_embeddings(model, testX)

# Save embeddings and labels
savez_compressed('5-celebrity-faces-embeddings.npz', train_embeddings, trainy, test_embeddings, testy)
print("Embeddings saved successfully. Shapes: ", train_embeddings.shape, test_embeddings.shape)
