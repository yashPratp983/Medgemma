import tensorflow as tf

SKIN_CLASSES = [
    "acanthosis nigricans", "acne", "acne vulgaris",
    "acquired autoimmune bullous diseaseherpes gestationis",
    "acrodermatitis enteropathica", "actinic keratosis",
    "allergic contact dermatitis", "aplasia cutis", "basal cell carcinoma",
    "basal cell carcinoma morpheiform", "becker nevus", "behcets disease",
    "calcinosis cutis", "cheilitis", "congenital nevus", "dariers disease",
    "dermatofibroma", "dermatomyositis", "disseminated actinic porokeratosis",
    "drug eruption", "drug induced pigmentary changes", "dyshidrotic eczema",
    "eczema", "ehlers danlos syndrome", "epidermal nevus",
    "epidermolysis bullosa", "erythema annulare centrifigum",
    "erythema elevatum diutinum", "erythema multiforme", "erythema nodosum",
    "factitial dermatitis", "fixed eruptions", "folliculitis", "fordyce spots",
    "granuloma annulare", "granuloma pyogenic", "hailey hailey disease",
    "halo nevus", "hidradenitis", "ichthyosis vulgaris",
    "incontinentia pigmenti", "juvenile xanthogranuloma", "kaposi sarcoma",
    "keloid", "keratosis pilaris", "langerhans cell histiocytosis",
    "lentigo maligna", "lichen amyloidosis", "lichen planus", "lichen simplex",
    "livedo reticularis", "lupus erythematosus", "lupus subacute",
    "lyme disease", "lymphangioma", "malignant melanoma", "melanoma", "milia",
    "mucinosis", "mucous cyst", "mycosis fungoides", "myiasis",
    "naevus comedonicus", "necrobiosis lipoidica", "nematode infection",
    "neurodermatitis", "neurofibromatosis", "neurotic excoriations",
    "neutrophilic dermatoses", "nevocytic nevus", "nevus sebaceous of jadassohn",
    "papilomatosis confluentes and reticulate", "paronychia", "pediculosis lids",
    "perioral dermatitis", "photodermatoses", "pilar cyst", "pilomatricoma",
    "pityriasis lichenoides chronica", "pityriasis rosea",
    "pityriasis rubra pilaris", "porokeratosis actinic",
    "porokeratosis of mibelli", "porphyria", "port wine stain",
    "prurigo nodularis", "psoriasis", "pustular psoriasis",
    "pyogenic granuloma", "rhinophyma", "rosacea", "sarcoidosis", "scabies",
    "scleroderma", "scleromyxedema", "seborrheic dermatitis",
    "seborrheic keratosis", "solid cystic basal cell carcinoma",
    "squamous cell carcinoma", "stasis edema", "stevens johnson syndrome",
    "striae", "sun damaged skin", "superficial spreading melanoma ssm",
    "syringoma", "telangiectases", "tick bite", "tuberous sclerosis",
    "tungiasis", "urticaria", "urticaria pigmentosa", "vitiligo", "xanthomas",
    "xeroderma pigmentosum",
]


def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    return tf.expand_dims(image, axis=0)


def predict(keras_model, image_bytes: bytes) -> tuple[str, float]:
    """Return (class_name, confidence)."""
    tensor = preprocess_image(image_bytes)
    predictions: list[float] = keras_model.predict(tensor)[0].tolist()
    max_idx = predictions.index(max(predictions))
    return SKIN_CLASSES[max_idx], round(predictions[max_idx], 4)
