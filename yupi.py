# Create feature extractor
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)


#explanation 
# Imagine you have a magical machine that looks at pictures and tells you important details about them without actually naming what the picture is. Here's what the code is doing:

# MobileNetV2 as a Feature Finder:
# Think of MobileNetV2 like a superhero with a special power—it can see and understand lots of different things in a picture. In our code, we're using MobileNetV2 that has already learned from many images. However, we don't want it to say "this is a cat" or "this is a car." We only want it to point out the interesting parts of the image.

# No Top Layer (include_top=False):
# Normally, MobileNetV2 would have a final part that tells you the exact name of what it sees. But here, we remove that part because we only need the details or "features" of the image, not the final answer.

# Global Average Pooling Layer:
# After the machine finds all these details, we need to turn them into one neat list (or vector) of numbers. The Global Average Pooling layer does that—it takes all the important details and averages them into one single list. Imagine it like gathering many scattered puzzle pieces into one clear picture summary.

# Feature Extractor Model:
# Finally, we create a new model called feature_extractor. This model takes an image as input and gives you that neat list of numbers (features). You can think of it as a special tool that, when you give it a picture, it tells you, "These are the important details about your picture."

synatax summary

# Let's break down the syntax step by step:

# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# MobileNetV2(...): This calls the MobileNetV2 function (or constructor) to create a model.
# weights='imagenet': This tells the model to load pre-trained weights from the ImageNet dataset.
# include_top=False: By setting this to False, we're not including the very last layers that normally classify images into categories. We only want the part that extracts features.
# input_shape=(224, 224, 3): This sets the expected shape of the input images. Here, images are 224 pixels wide, 224 pixels tall, and have 3 color channels (red, green, blue).
# The result is stored in the variable base_model.
# x = GlobalAveragePooling2D()(base_model.output)

# GlobalAveragePooling2D(): This creates an instance of the global average pooling layer.
# (base_model.output): The layer is then immediately applied to base_model.output, which is the output (feature maps) from the MobileNetV2 model.
# Think of this as taking all the tiny details (numbers in the feature maps) and averaging them into one simple list. The result is stored in the variable x.
# feature_extractor = Model(inputs=base_model.input, outputs=x)

# Model(...): This is a function that creates a new model.
# inputs=base_model.input: It takes the original input of MobileNetV2 (the image you provide).
# outputs=x: It uses the output x from the pooling layer as the model's final output.
# The new model is assigned to the variable feature_extractor.
# In summary, the syntax creates a new model that:

# Starts with MobileNetV2 to analyze images,
# Uses a global average pooling layer to simplify the information into a neat list,
# And builds a final model that you can use to extract these features from any image you pass in.


model = None
le = LabelEncoder()

#1 model = None

# This line creates a variable named model and sets it to None.
# Think of it like an empty box. We don't have our SVM classifier ready yet, so we leave this box empty for now. Later in the code, we'll put the actual SVM model into this variable.

#2 le = LabelEncoder()

# This line creates a new instance (an object) of the LabelEncoder class and assigns it to the variable le.
# The LabelEncoder is used to convert human-readable labels (like folder names or categories) into numbers. Computers work better with numbers, so this step makes it easier for the model to understand the labels during training.




def extract_features(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = feature_extractor.predict(x)
    return features.flatten()



# The function image.load_img is used to load an image file from your disk into your Python program. It does a couple of key things:

# Reading the Image: It opens the image file from the given file path.
# Resizing: With the parameter target_size=IMG_SIZE, it resizes the image to the dimensions specified by IMG_SIZE (in this case, 224x224 pixels). This ensures all images are the same size, which is important for the model to process them correctly.
# In simple terms, it takes an image file (like a photo on your computer) and prepares it (by loading and resizing) so that it can be further processed, for example, by a neural network.
# img = image.load_img(img_path, target_size=IMG_SIZE)
# What's happening?
# The function takes an image file path (img_path) and loads the image from that location. It also resizes the image to a specific size (IMG_SIZE), so all images are the same size.
# Imagine it like:
# You're taking a photo and then using a tool to make sure it fits perfectly into a standard picture frame.
# Conversion to Array:


x = image.img_to_array(img)
# What's happening?
# The image is turned into a numerical array. Computers understand numbers better than pictures.
# Imagine it like:
# Converting a drawing into a list of numbers that describe the colors and details in the picture.
# Adding a Batch Dimension:


x = np.expand_dims(x, axis=0)
# What's happening?
# The image array is reshaped by adding an extra dimension so it looks like a batch of images (even if it's just one). This is needed because the model expects input data in batches.
# Imagine it like:
# Even if you have one cookie, you still put it on a tray that can hold several cookies, so the cookie machine knows how to handle it.
# Preprocessing the Image:


x = preprocess_input(x)
# What's happening?
# The image data is modified (preprocessed) in a way that MobileNetV2 expects. This could mean normalizing the numbers or doing other adjustments.
# Imagine it like:
# Giving your drawing a little touch-up before showing it to an art expert.
# Extracting Features:


features = feature_extractor.predict(x)
# What's happening?
# The preprocessed image is passed through the feature extractor (which is our modified MobileNetV2 model). The model looks at the image and outputs a list of important features (or clues) about the image.
# Imagine it like:
# A super-smart camera that doesn’t just see the whole picture but picks out details like shapes, textures, and patterns.
# Flattening the Feature Vector:


return features.flatten()
# What's happening?
# The list of features is "flattened," which means it’s converted into a one-dimensional list (a single row of numbers). This makes it easier to work with in later steps, like training a classifier.
# Imagine it like:
# Taking a stack of papers and turning it into a single long list of notes, so everything is neat and in one line.
# In summary:
# The function extract_features takes a picture file, resizes it, converts it into numbers, preps it the way our smart model likes, uses the model to pick out key details, and finally gives you those details as one neat list of numbers. This list of numbers represents the essential features of the image that can be used later for tasks like classifying or comparing images.