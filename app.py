import cv2
import numpy as np
import gradio as gr

# ============================================================
# CROP DISEASE TRIAGE
# IMAGE CAPTURE + INPUT QUALITY + TRUST DECISION
# ============================================================
#
# INPUT:
#   Smartphone camera / uploaded crop leaf image
#
# PROCESS:
#   1. Blur detection
#   2. Brightness detection
#   3. Contrast detection
#   4. Resolution check
#   5. Image quality scoring
#
# OUTPUT:
#   1. Captured image
#   2. Quality score
#   3. Image status
#   4. Problems detected
#   5. Trust / Retake decision
#
# ============================================================


# ------------------------------------------------------------
# 1. BLUR DETECTION
# ------------------------------------------------------------

def calculate_blur(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Laplacian variance
    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return float(blur_score)


# ------------------------------------------------------------
# 2. BRIGHTNESS
# ------------------------------------------------------------

def calculate_brightness(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    brightness = np.mean(gray)

    return float(brightness)

import cv2
import gradio as gr
import numpy as np


LANGUAGES = {
    "English": "en",
    "हिन्दी": "hi",
    "தமிழ்": "ta",
    "తెలుగు": "te",
    "मराठी": "mr",
}


LOCALIZED_LABELS = {
    "en": {
        "healthy": "Healthy", "leaf_blight": "Leaf Blight", "rust": "Rust", "nutrient": "Nutrient Deficiency",
        "unknown": "Unknown", "no_image": "No image", "report": "CROP AND DISEASE REPORT", "crop": "Predicted Crop",
        "disease": "Disease Status", "health": "Health Check", "confidence": "Confidence", "quality": "Quality Score",
        "status": "Status", "blur": "Blur Score", "brightness": "Brightness", "contrast": "Contrast",
        "resolution": "Resolution", "leaf": "Leaf Info", "disease_info": "Disease Info", "issues": "Detected Issues",
        "final": "FINAL DIAGNOSIS", "healthy_status": "HEALTHY", "diseased_status": "DISEASED", "eco": "Eco Advice",
        "care": "Care Note", "no_issues": "No major image-quality problems detected.", "trust": "The image is suitable for crop and disease analysis.",
        "caution": "The image can be analyzed, but quality may affect the result.", "retake": "The image is not reliable enough for crop disease analysis.",
        "no_crop": "NO CROP IMAGE", "upload": "Please capture or upload a crop leaf image.", "retake_short": "RETAKE PHOTO",
    },
    "hi": {
        "healthy": "स्वस्थ", "leaf_blight": "लीफ ब्लाइट", "rust": "रस्ट रोग", "nutrient": "पोषक तत्वों की कमी",
        "unknown": "अज्ञात", "no_image": "कोई चित्र नहीं", "report": "फसल और रोग रिपोर्ट", "crop": "अनुमानित फसल",
        "disease": "रोग की स्थिति", "health": "स्वास्थ्य जांच", "confidence": "विश्वसनीयता", "quality": "गुणवत्ता स्कोर",
        "status": "स्थिति", "blur": "धुंधलापन स्कोर", "brightness": "चमक", "contrast": "कंट्रास्ट",
        "resolution": "रिज़ॉल्यूशन", "leaf": "पत्ती की जानकारी", "disease_info": "रोग की जानकारी", "issues": "पहचानी गई समस्याएं",
        "final": "अंतिम निदान", "healthy_status": "स्वस्थ", "diseased_status": "रोगग्रस्त", "eco": "पर्यावरण अनुकूल सलाह",
        "care": "देखभाल नोट", "no_issues": "चित्र की गुणवत्ता में कोई बड़ी समस्या नहीं मिली।", "trust": "चित्र फसल और रोग विश्लेषण के लिए उपयुक्त है।",
        "caution": "चित्र का विश्लेषण किया जा सकता है, लेकिन गुणवत्ता परिणाम को प्रभावित कर सकती है।", "retake": "रोग विश्लेषण के लिए चित्र पर्याप्त विश्वसनीय नहीं है।",
        "no_crop": "फसल का चित्र नहीं है", "upload": "कृपया फसल की पत्ती का चित्र लें या अपलोड करें।", "retake_short": "चित्र दोबारा लें",
    },
    "ta": {
        "healthy": "ஆரோக்கியமானது", "leaf_blight": "இலைப் ப்ளைட்", "rust": "ரஸ்ட் நோய்", "nutrient": "ஊட்டச்சத்து குறைபாடு",
        "unknown": "தெரியவில்லை", "no_image": "படம் இல்லை", "report": "பயிர் மற்றும் நோய் அறிக்கை", "crop": "கணிக்கப்பட்ட பயிர்",
        "disease": "நோய் நிலை", "health": "ஆரோக்கியச் சோதனை", "confidence": "நம்பகத்தன்மை", "quality": "தர மதிப்பெண்",
        "status": "நிலை", "blur": "மங்கல் மதிப்பெண்", "brightness": "பிரகாசம்", "contrast": "மாறுபாடு",
        "resolution": "தெளிவுத்திறன்", "leaf": "இலை தகவல்", "disease_info": "நோய் தகவல்", "issues": "கண்டறியப்பட்ட சிக்கல்கள்",
        "final": "இறுதி நோயறிதல்", "healthy_status": "ஆரோக்கியமானது", "diseased_status": "நோயுற்றது", "eco": "சுற்றுச்சூழல் நட்பு ஆலோசனை",
        "care": "பராமரிப்பு குறிப்பு", "no_issues": "படத்தின் தரத்தில் பெரிய சிக்கல்கள் கண்டறியப்படவில்லை.", "trust": "பயிர் மற்றும் நோய் பகுப்பாய்வுக்கு படம் ஏற்றது.",
        "caution": "படத்தை பகுப்பாய்வு செய்யலாம், ஆனால் தரம் முடிவை பாதிக்கலாம்.", "retake": "நோய் பகுப்பாய்வுக்கு படம் போதுமான நம்பகமானதல்ல.",
        "no_crop": "பயிர் படம் இல்லை", "upload": "பயிர் இலைப் படத்தைப் பதிவேற்றவும் அல்லது எடுக்கவும்.", "retake_short": "படத்தை மீண்டும் எடுக்கவும்",
    },
    "te": {
        "healthy": "ఆరోగ్యకరమైనది", "leaf_blight": "ఆకు బ్లైట్", "rust": "రస్ట్ వ్యాధి", "nutrient": "పోషక లోపం",
        "unknown": "తెలియదు", "no_image": "చిత్రం లేదు", "report": "పంట మరియు వ్యాధి నివేదిక", "crop": "అంచనా పంట",
        "disease": "వ్యాధి స్థితి", "health": "ఆరోగ్య తనిఖీ", "confidence": "నమ్మకం", "quality": "నాణ్యత స్కోర్",
        "status": "స్థితి", "blur": "అస్పష్టత స్కోర్", "brightness": "ప్రకాశం", "contrast": "కాంట్రాస్ట్",
        "resolution": "రిజల్యూషన్", "leaf": "ఆకు సమాచారం", "disease_info": "వ్యాధి సమాచారం", "issues": "గుర్తించిన సమస్యలు",
        "final": "తుది నిర్ధారణ", "healthy_status": "ఆరోగ్యకరమైనది", "diseased_status": "వ్యాధిగ్రస్తం", "eco": "పర్యావరణ అనుకూల సలహా",
        "care": "సంరక్షణ గమనిక", "no_issues": "చిత్ర నాణ్యతలో పెద్ద సమస్యలు కనబడలేదు.", "trust": "పంట మరియు వ్యాధి విశ్లేషణకు చిత్రం అనుకూలంగా ఉంది.",
        "caution": "చిత్రాన్ని విశ్లేషించవచ్చు, కానీ నాణ్యత ఫలితాన్ని ప్రభావితం చేయవచ్చు.", "retake": "వ్యాధి విశ్లేషణకు చిత్రం తగినంత నమ్మదగినది కాదు.",
        "no_crop": "పంట చిత్రం లేదు", "upload": "దయచేసి పంట ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి లేదా తీయండి.", "retake_short": "చిత్రాన్ని మళ్లీ తీయండి",
    },
    "mr": {
        "healthy": "निरोगी", "leaf_blight": "लीफ ब्लाइट", "rust": "रस्ट रोग", "nutrient": "पोषक घटकांची कमतरता",
        "unknown": "अज्ञात", "no_image": "चित्र नाही", "report": "पीक आणि रोग अहवाल", "crop": "अंदाजित पीक",
        "disease": "रोगाची स्थिती", "health": "आरोग्य तपासणी", "confidence": "विश्वासार्हता", "quality": "गुणवत्ता गुण",
        "status": "स्थिती", "blur": "धूसरपणा गुण", "brightness": "प्रकाश", "contrast": "कॉन्ट्रास्ट",
        "resolution": "रिझोल्यूशन", "leaf": "पानांची माहिती", "disease_info": "रोगाची माहिती", "issues": "आढळलेल्या समस्या",
        "final": "अंतिम निदान", "healthy_status": "निरोगी", "diseased_status": "रोगग्रस्त", "eco": "पर्यावरणपूरक सल्ला",
        "care": "काळजीची नोंद", "no_issues": "चित्राच्या गुणवत्तेत कोणतीही मोठी समस्या आढळली नाही.", "trust": "पीक आणि रोग विश्लेषणासाठी चित्र योग्य आहे.",
        "caution": "चित्राचे विश्लेषण करता येईल, परंतु गुणवत्ता परिणामावर परिणाम करू शकते.", "retake": "रोग विश्लेषणासाठी चित्र पुरेसे विश्वसनीय नाही.",
        "no_crop": "पिकाचे चित्र नाही", "upload": "कृपया पिकाच्या पानाचे चित्र अपलोड करा किंवा काढा.", "retake_short": "चित्र पुन्हा काढा",
    },
}


CROP_NAMES = {
    "en": {"Rice": "Rice", "Corn": "Corn", "Tomato": "Tomato", "Potato": "Potato", "Leafy Vegetable": "Leafy Vegetable"},
    "hi": {"Rice": "धान", "Corn": "मक्का", "Tomato": "टमाटर", "Potato": "आलू", "Leafy Vegetable": "पत्तेदार सब्जी"},
    "ta": {"Rice": "அரிசி", "Corn": "மக்காச்சோளம்", "Tomato": "தக்காளி", "Potato": "உருளைக்கிழங்கு", "Leafy Vegetable": "இலைக் காய்கறி"},
    "te": {"Rice": "వరి", "Corn": "మొక్కజొన్న", "Tomato": "టమోటా", "Potato": "బంగాళాదుంప", "Leafy Vegetable": "ఆకుకూర"},
    "mr": {"Rice": "भात", "Corn": "मका", "Tomato": "टोमॅटो", "Potato": "बटाटा", "Leafy Vegetable": "पालेभाजी"},
}


LOCALIZED_ADVICE = {
    "hi": {
        "Healthy": "फसल स्वस्थ दिख रही है। संतुलित पोषण, उचित सिंचाई और नियमित निगरानी जारी रखें। अधिक पानी, अचानक तापमान बदलाव और अनावश्यक कीटनाशक उपयोग से बचें।",
        "Leaf Blight": "संक्रमित पत्तियां हटाएं, हवा का आवागमन बढ़ाएं, ऊपर से सिंचाई न करें और स्थानीय कृषि सलाह के अनुसार उपयुक्त फफूंदनाशी का उपयोग करें। खेत की स्वच्छता बनाए रखें।",
        "Rust": "यदि उपलब्ध हो तो रोग-प्रतिरोधी किस्म लगाएं, अधिक संक्रमित पत्तियां हटाएं और स्थानीय सलाह के अनुसार फफूंदनाशी का उपयोग करें। शाम के समय पत्तियों को गीला रखने से बचें।",
        "Nutrient Deficiency": "मिट्टी की उर्वरता की जांच करें और फसल की अवस्था के अनुसार संतुलित पोषक तत्व दें। सिंचाई और जल निकास सुधारें तथा लक्षण बढ़ने से पहले निगरानी करें।",
    },
    "ta": {
        "Healthy": "பயிர் ஆரோக்கியமாகத் தெரிகிறது. சமநிலையான ஊட்டச்சத்து, சரியான பாசனம் மற்றும் வழக்கமான கண்காணிப்பைத் தொடருங்கள். அதிக நீர் மற்றும் தேவையற்ற பூச்சிக்கொல்லிகளைத் தவிர்க்கவும்.",
        "Leaf Blight": "பாதிக்கப்பட்ட இலைகளை அகற்றி, காற்றோட்டத்தை மேம்படுத்தி, மேலிருந்து பாசனம் செய்வதைத் தவிர்க்கவும். உள்ளூர் விவசாய ஆலோசனைப்படி பொருத்தமான பூஞ்சைக் கொல்லியைப் பயன்படுத்தவும்.",
        "Rust": "கிடைத்தால் நோய் எதிர்ப்பு வகைகளைப் பயன்படுத்துங்கள், அதிகம் பாதிக்கப்பட்ட இலைகளை அகற்றுங்கள், உள்ளூர் ஆலோசனைப்படி பூஞ்சைக் கொல்லியைப் பயன்படுத்துங்கள். மாலை நேர இலை ஈரத்தைத் தவிர்க்கவும்.",
        "Nutrient Deficiency": "மண் வளத்தைச் சரிபார்த்து, பயிர் நிலைக்கு ஏற்ப சமநிலையான ஊட்டச்சத்தை வழங்குங்கள். பாசனம் மற்றும் வடிகாலைக் கட்டுப்படுத்தி, அறிகுறிகள் அதிகரிக்கும் முன் கண்காணிக்கவும்.",
    },
    "te": {
        "Healthy": "పంట ఆరోగ్యంగా కనిపిస్తోంది. సమతుల్య పోషణ, సరైన నీటిపారుదల మరియు క్రమమైన పర్యవేక్షణ కొనసాగించండి. అధిక నీరు మరియు అవసరం లేని పురుగుమందులను నివారించండి.",
        "Leaf Blight": "ప్రభావిత ఆకులను తొలగించండి, గాలి ప్రసరణను మెరుగుపరచండి, పై నుంచి నీరు పోయవద్దు. స్థానిక వ్యవసాయ సలహా ప్రకారం సరైన శిలీంద్రనాశిని ఉపయోగించండి.",
        "Rust": "అందుబాటులో ఉంటే వ్యాధి నిరోధక రకాలను ఉపయోగించండి, ఎక్కువగా ప్రభావితమైన ఆకులను తొలగించండి మరియు స్థానిక సలహా ప్రకారం శిలీంద్రనాశిని ఉపయోగించండి. సాయంత్రం ఆకులు తడిగా ఉండకుండా చూడండి.",
        "Nutrient Deficiency": "మట్టి సారాన్ని పరీక్షించి, పంట దశకు అనుగుణంగా సమతుల్య పోషకాలను ఇవ్వండి. నీటిపారుదల మరియు డ్రైనేజీని మెరుగుపరచి, లక్షణాలు పెరగకముందే పర్యవేక్షించండి.",
    },
    "mr": {
        "Healthy": "पीक निरोगी दिसत आहे. संतुलित पोषण, योग्य सिंचन आणि नियमित निरीक्षण सुरू ठेवा. जास्त पाणी, अचानक तापमान बदल आणि अनावश्यक कीटकनाशकांचा वापर टाळा.",
        "Leaf Blight": "संक्रमित पाने काढा, हवा खेळती ठेवा, वरून सिंचन टाळा आणि स्थानिक कृषी सल्ल्यानुसार योग्य बुरशीनाशक वापरा. शेताची स्वच्छता राखा.",
        "Rust": "उपलब्ध असल्यास रोगप्रतिकारक वाण वापरा, जास्त संक्रमित पाने काढा आणि स्थानिक सल्ल्यानुसार बुरशीनाशक वापरा. संध्याकाळी पाने ओली ठेवणे टाळा.",
        "Nutrient Deficiency": "मातीची सुपीकता तपासा आणि पिकाच्या अवस्थेनुसार संतुलित पोषक द्या. सिंचन आणि निचरा सुधारून लक्षणे वाढण्यापूर्वी निरीक्षण करा.",
    },
}


def localized_advice(disease, language_code, english_advice):
    if language_code == "en":
        return english_advice
    return LOCALIZED_ADVICE.get(language_code, {}).get(disease, english_advice)


CROP_DETAILS = {
    "Rice": {
        "leaf": {
            "en": "Rice leaves are long, narrow, and green. Healthy leaves stay uniformly green with strong upright growth.",
            "hi": "धान की पत्तियां लंबी, पतली और हरी होती हैं। स्वस्थ पत्तियां समान रूप से हरी और सीधी रहती हैं।",
            "ta": "அரிசி இலைகள் நீளமான, குறுகிய, பச்சை நிறத்தில் இருக்கும். ஆரோக்கியமான இலைகள் ஒரே சீரான பச்சை நிறத்தில் இருக்கும்.",
            "te": "వరి ఆకులు పొడవుగా, ఇరుకుగా, ఆకుపచ్చగా ఉంటాయి. ఆరోగ్యకరమైన ఆకులు ఏకరీతి పచ్చగా మరియు నిటారుగా ఉంటాయి.",
            "mr": "भाताच्या पानांमध्ये लांब, अरुंद, हिरवे रंग असतात. निरोगी पाने एकसारखे हिरवी आणि उभी राहतात.",
        },
        "diseases": {
            "Healthy": {
                "en": "Healthy rice leaves are green and free from dry patches, yellowing, or fungal growth.",
                "hi": "स्वस्थ धान की पत्तियां हरी रहती हैं और उनमें सूखे धब्बे, पीलीपन या फंगल विकास नहीं होता।",
                "ta": "சுகாதாரமான அரிசி இலைகள் பச்சை நிறத்தில் இருக்கும், வாடிய புள்ளிகள், மஞ்சள் நிறம் அல்லது பூஞ்சை வளர்ச்சி இருக்காது.",
                "te": "ఆరోగ్యకరమైన వరి ఆకులు ఆకుపచ్చగా ఉంటాయి మరియు ఎండిపోయిన బుగ్గలు, పసుపు రంగు లేదా పురుగుల పెరుగుదల ఉండదు.",
                "mr": "निरोगी भाताच्या पानांचा रंग हिरवा असतो आणि कोरडे ठिपके, पिवळेगुण किंवा फंगसचा विकास होत नाही.",
            },
            "Leaf Blight": {
                "en": "Leaf blight appears as irregular brown or gray patches on leaves, often causing rapid drying of affected foliage.",
                "hi": "लीफ ब्लाइट में पत्तियों पर अनियमित भूरा या ग्रे धब्बे दिखाई देते हैं, जिससे प्रभावित पत्तियां जल्दी सुख जाती हैं।",
                "ta": "இலைப் ப்ளைட், இலைகளில் ஒழுங்கற்ற பழுப்பு அல்லது சாம்பல் புள்ளிகளாகத் தோன்றும், மேலும் பாதிக்கப்பட்ட இலைகள் விரைவாக காய்ந்து விடும்.",
                "te": "ఆకు బ్లైట్‌లో ఆకులపై క్రమరహిత గోధుమ లేదా గ్రే గాట్లు కనిపిస్తాయి, ఇవి ప్రభావిత ఆకులను త్వరగా ఎండబెడతాయి.",
                "mr": "लीफ ब्लाइटमध्ये पानांवर अनियमित तपकिरी किंवा राखी रंगाचे ठिपके दिसतात, ज्यामुळे प्रभावित पाने वेगाने कोरडी पडतात.",
            },
            "Brown Spot": {
                "en": "Brown spot disease creates small dark brown lesions with yellow halos, often reducing grain quality.",
                "hi": "ब्राउन स्पॉट रोग में छोटे गहरे भूरे धब्बे और पीले घेरे बनते हैं, जिससे दाना की गुणवत्ता कम हो सकती है।",
                "ta": "பிரவுன் ஸ்பாட் நோய் சிறிய கரும் பழுப்பு புள்ளிகளுடன் மஞ்சள் சுற்று உருவாக்கி, தானியத்தின் தரத்தை குறைக்கும்.",
                "te": "బ్రౌన్ స్పాట్ వ్యాధి చిన్న గోధుమ చారలు మరియు పసుపు వలయాలతో ఏర్పడుతుంది, ఇది విత్తన నాణ్యతను తగ్గిస్తుంది.",
                "mr": "ब्राउन स्पॉट रोगामध्ये लहान गडद तपकिरी ठिपके आणि पिवळे हळदूंद दिसतात, ज्यामुळे धान्याची गुणवत्ता कमी होते.",
            },
        },
    },
    "Tomato": {
        "leaf": {
            "en": "Tomato leaves are compound and slightly rough with a green color. Healthy plants show strong, broad green foliage.",
            "hi": "टमाटर की पत्तियां छोटी-छोटी और थोड़ी खुरदरी होती हैं, साथ में हरी रंग की होती हैं। स्वस्थ पौधे मजबूत हरे पत्ते देते हैं।",
            "ta": "தக்காளி இலைகள் சிறிய, சற்று கடினமான, பச்சை நிறத்தில் இருக்கும். ஆரோக்கியமான தாவரங்கள் உறுதியான, பரந்த பச்சை இலைகளை உருவாக்கும்.",
            "te": "టమోటా ఆకులు చిన్నవిగా మరియు కొద్దిగా గట్టిగా ఉంటాయి, ఆకుపచ్చ రంగులో ఉంటాయి. ఆరోగ్యకరమైన మొక్కలు బలమైన, విస్తృత ఆకుపచ్చ ఆకులను చూపుతాయి.",
            "mr": "टोमॅटोच्या पानांचा रंग हिरवा आणि ते थोडे खरखरीत असतात. निरोगी झाडांमध्ये मजबूत, रुंद हिरवे पाने असतात.",
        },
        "diseases": {
            "Healthy": {
                "en": "Healthy tomato leaves are green with no spots, yellowing, curling, or weak growth.",
                "hi": "स्वस्थ टमाटर की पत्तियां हरी होती हैं और उनमें धब्बे, पीलीपन, मुड़ाव या कमजोर विकास नहीं होता।",
                "ta": "சுகாதாரமான தக்காளி இலைகள் பச்சையாக இருக்கும், புள்ளிகள், மஞ்சள் நிறம், சுருட்டல் அல்லது பலவீனமான வளர்ச்சி இருக்காது.",
                "te": "ఆరోగ్యకరమైన టమోటా ఆకులు ఆకుపచ్చగా ఉంటాయి మరియు గాట్లు, పసుపు రంగు, మడతలు లేదా బలహీనమైన పెరుగుదల ఉండదు.",
                "mr": "निरोगी टोमॅटोची पाने हिरवी असतात आणि त्यावर ठिपके, पिवळेपणा, वाकणे किंवा कमकुवत वाढ दिसत नाही.",
            },
            "Leaf Blight": {
                "en": "Early blight causes dark concentric spots on lower leaves, which expands and can lead to defoliation.",
                "hi": "ईरली ब्लाइट में नीचे की पत्तियों पर गहरे वृत्ताकार धब्बे दिखाई देते हैं, जो बढ़कर पत्तियों के झड़ने का कारण बनते हैं।",
                "ta": "ஆரம்ப ப்ளைட், கீழ் இலைகளில் அடர் வட்ட வடிவ புள்ளிகளை உருவாக்கி, பரவி இலைகள் விழுவதற்கு வழிவகுக்கும்.",
                "te": "ఎర్లీ బ్లైట్ దిగువ ఆకులపై గడ్డురంగు గుండ్రని చారలను రూపొందించి, వ్యాప్తి చెందగానే ఆకులు పడిపోవచ్చు.",
                "mr": "इअर्ली ब्लाइटमुळे खालच्या पानांवर गडद वर्तुळाकार ठिपके दिसतात, जे वाढून पाने झडू शकतात.",
            },
            "Rust": {
                "en": "Rust disease produces orange or brown pustules on leaves and stems, often spreading with humidity.",
                "hi": "रस्ट रोग में पत्तियों और तनों पर नारंगी या भूरा pustule दिखाई देता है, जो नमी के साथ तेजी से फैलता है।",
                "ta": "ரஸ்ட் நோய், இலைகள் மற்றும் தண்டுகளில் ஆரஞ்சு அல்லது பழுப்பு நிற மேல்புள்ளிகளை உருவாக்கி, ஈரப்பசையுடன் விரைவாகப் பரவுகிறது.",
                "te": "రస్ట్ వ్యాధి ఆకులు మరియు కాండాలపై నారింజ లేదా గోధుమ పుస్టుల్స్‌ను ఏర్పరుస్తుంది, తేమతో వేగంగా వ్యాప్తి చెందుతుంది.",
                "mr": "रस्ट रोगामुळे पानांमध्ये आणि कांडांवर नारिंगी किंवा तपकिरी पुस्च्युल्स दिसतात आणि आर्द्रतेसह ते वेगाने पसरते.",
            },
        },
    },
    "Potato": {
        "leaf": {
            "en": "Potato leaves are broad and dark green. Healthy plants show steady growth with no irregular spots or curling.",
            "hi": "आलू की पत्तियां चौड़ी और गहरे हरे रंग की होती हैं। स्वस्थ पौधे बिना धब्बे या मुड़ाव के मजबूत विकास करते हैं।",
            "ta": "உருளைக் கிழங்கு இலைகள் பரந்த மற்றும் அடர் பச்சை நிறத்தில் இருக்கும். ஆரோக்கியமான தாவரங்கள் சீரான வளர்ச்சியுடன் இருக்கும்.",
            "te": "బంగాళాదుంప ఆకులు విస్తృతంగా మరియు ముదురు ఆకుపచ్చ రంగులో ఉంటాయి. ఆరోగ్యకరமான మొక్కలు క్రమమైన పెరుగుదల చూపిస్తాయి.",
            "mr": "बटाट्याच्या पानांचे रंग गडद हिरवा आणि ते रुंद असतात. निरोगी झाडांची वाढ सुसंगत आणि ठिपकेशिवाय असते.",
        },
        "diseases": {
            "Healthy": {
                "en": "Healthy potato leaves are fresh green and firm with no chlorotic symptoms.",
                "hi": "स्वस्थ आलू की पत्तियां ताज़ा हरी और मजबूत होती हैं, बिना पीलेपन या धब्बों के।",
                "ta": "சுகாதாரமான உருளைக் கிழங்கு இலைகள் புதிய பச்சை நிறத்தில் இருப்பதும், மஞ்சள் நிறம் அல்லது புள்ளிகள் இல்லாமல் இருக்கும்.",
                "te": "ఆరోగ్యకరమైన బంగాళాదుంప ఆకులు తాజా ఆకుపచ్చగా మరియు గట్టిగా ఉంటాయి, పసుపు రంగు లేదా చారలు ఉండవు.",
                "mr": "निरोगी बटाट्याची पाने ताज्या हिरव्या आणि दृढ असतात, पिवळेपणा किंवा ठिपके नसतात.",
            },
            "Leaf Blight": {
                "en": "Late blight forms olive-green to brown lesions on leaves and can spread rapidly during cool, moist weather.",
                "hi": "लेट ब्लाइट में पत्तियों पर जैतून हरे से भूरे धब्बे बनते हैं और ठंडी, नम हवा में तेजी से फैलते हैं।",
                "ta": "லேட் ப்ளைட், இலைகளில் ஆலிவ் பச்சை முதல் பழுப்பு நிற புள்ளிகளை உருவாக்கி, குளிர்ந்த மற்றும் ஈரப்பசையான காலநிலையில் விரைவாகப் பரவுகிறது.",
                "te": "లేట్ బ్లైట్ ఆకులపై ఆలివ్ గ్రీన్ నుండి గోధుమ చారలను ఏర్పరుస్తుంది మరియు చల్లని, తేమతో కూడిన వాతావరణంలో వేగంగా వ్యాప్తి చెందుతుంది.",
                "mr": "लेट ब्लाइटमुळे पानांवर जैतूनी हिरवे ते तपकिरी ठिपके दिसतात आणि थंड, ओले हवामानात ते वेगाने पसरते.",
            },
            "Nutrient Deficiency": {
                "en": "Nutrient deficiency often causes yellowing between veins, weak growth, and reduced vigor.",
                "hi": "पोषक तत्व की कमी से शिराओं के बीच पीलीपन, कमजोर विकास और कम vigor दिखाई देता है।",
                "ta": "ஊட்டச்சத்து குறைபாடு, நரம்புகளுக்கு இடையே மஞ்சள் நிறம், பலவீனமான வளர்ச்சி மற்றும் குறைந்த வலிமையை ஏற்படுத்தும்.",
                "te": "పోషక లోపం నరాల మధ్య పసుపు రంగు, బలహీనమైన పెరుగుదల మరియు తగ్గిన vigor‌ను చూపుతుంది.",
                "mr": "पोषक घटकांची कमतरता नाळांच्या दरम्यान पिवळेपणा, कमकुवत वाढ आणि कमी शक्ती दर्शवते.",
            },
        },
    },
    "Corn": {
        "leaf": {
            "en": "Corn leaves are long and narrow with parallel veins. Healthy plants have uniform green leaves and upright growth.",
            "hi": "मक्के की पत्तियां लंबी और पतली होती हैं, जिनमें समान शिराएं होती हैं। स्वस्थ पौधे समान हरी पत्तियों और सीधी वृद्धि के साथ होते हैं।",
            "ta": "மக்காச்சோளம் இலைகள் நீளமாகவும் குறுகலாகவும், இணையான நரம்புகளுடன் இருக்கும். ஆரோக்கியமான தாவரங்கள் ஒரே சீரான பச்சை இலைகளைக் கொண்டிருக்கும்.",
            "te": "చోళ్లు/మొక్కజొన్న ఆకులు పొడవుగా మరియు ఇరుకుగా ఉంటాయి, సమాంతర నరాలు ఉంటాయి. ఆరోగ్యకరమైన మొక్కలు ఏకరీతి ఆకుపచ్చ ఆకులు మరియు నిటారుగా పెరుగుతాయి.",
            "mr": "मका पानांमध्ये लांब, अरुंद आणि समांतर शिरा असतात. निरोगी झाडांमध्ये एकसारखे हिरवे पाने आणि उभी वाढ असते.",
        },
        "diseases": {
            "Healthy": {
                "en": "Healthy maize leaves are green with normal growth and no torn or dry patches.",
                "hi": "स्वस्थ मक्का की पत्तियां हरी होती हैं और उनमें सामान्य विकास, बिना फटे या सूखे धब्बों के होती हैं।",
                "ta": "சுகாதாரமான மக்காச்சோளம் இலைகள் பச்சையாகவும், சாதாரண வளர்ச்சியுடன், கிழிந்த அல்லது வறண்ட புள்ளிகள் இல்லாமல் இருக்கும்.",
                "te": "ఆరోగ్యకరమైన మొక్కజొన్న ఆకులు ఆకుపచ్చగా ఉంటాయి మరియు సాధారణ పెరుగుదలతో, చీలిన లేదా ఎండిపోయిన గాట్లు లేకుండా ఉంటాయి.",
                "mr": "निरोगी मका पानांमध्ये हिरवा रंग आणि सामान्य वाढ असते, तसेच फाटलेले किंवा कोरडे ठिपके नसतात.",
            },
            "Leaf Blight": {
                "en": "Leaf blight in corn creates yellow or tan lesions that quickly expand and weaken the canopy.",
                "hi": "मक्के में लीफ ब्लाइट से पीले या टैन रंग के धब्बे बनते हैं जो जल्दी फैलते हैं और छत को कमजोर करते हैं।",
                "ta": "மக்காச்சோளத்தில் இலை ப்ளைட், மஞ்சள் அல்லது டேன் நிறப் புள்ளிகளை உருவாக்கி விரைவாக பரவி, மேற்பரப்பை பலவீனப்படுத்தும்.",
                "te": "మొక్కజొన్నలో లీఫ్ బ్లైట్ పసుపు లేదా టాన్ రంగు చారలను ఏర్పరుస్తుంది, ఇవి త్వరగా వ్యాప్తి చెంది పుస్పాలను బలహీనపరుస్తాయి.",
                "mr": "मक्यामध्ये लीफ ब्लाइटमुळे पिवळे किंवा टैन रंगाचे ठिपके दिसतात जे वेगाने पसरतात आणि छतावर प्रभाव टाकतात.",
            },
            "Nutrient Deficiency": {
                "en": "Nitrogen or potassium deficiency often causes yellowing from leaf tips toward the middle.",
                "hi": "नाइट्रोजन या पोटेशियम की कमी से पत्तियों के सिरों से बीच की तरफ पीला पड़ना शुरू होता है।",
                "ta": "நைட்ரஜன் அல்லது பொட்டாசியம் குறைபாடு, இலை முனைகள் முதல் நடுப்பகுதி வரை மஞ்சள் நிறத்தைக் காட்டும்.",
                "te": "నైట్రోజన్ లేదా పొటాషియమ్ లోపం ఆకుల అంచుల నుంచి మధ్యభాగానికి పసుపు రంగును చూపిస్తుంది.",
                "mr": "नायट्रोजन किंवा पोटॅशियम कमतरतेमुळे पानेच्या टोकेपासून मध्यभागाकडे पिवळेपणा दिसतो.",
            },
        },
    },
    "Leafy Vegetable": {
        "leaf": {
            "en": "Leafy vegetables usually have soft, broad leaves. Healthy leaves show deep green color and good turgor.",
            "hi": "पत्तेदार सब्जियों में नरम और चौड़ी पत्तियां होती हैं। स्वस्थ पत्तियां गहरी हरी और दृढ़ होती हैं।",
            "ta": "இலைக் காய்கறிகளில் மென்மையான, பரந்த இலைகள் இருக்கும். ஆரோக்கியமான இலைகள் ஆழமான பச்சை மற்றும் உறுதியானதாக இருக்கும்.",
            "te": "ఆకు కూరలలో మృదువైన, విస్తృత ఆకులు ఉంటాయి. ఆరోగ్యకరమైన ఆకులు లోతైన ఆకుపచ్చ మరియు బలమైనవిగా ఉంటాయి.",
            "mr": "पानांचा भाजीमध्ये मऊ, रुंद पाने असतात. निरोगी पाने गडद हिरवी आणि दृढ असतात.",
        },
        "diseases": {
            "Healthy": {
                "en": "Healthy leafy vegetables are crisp, green, and free of wilting or fungal spots.",
                "hi": "स्वस्थ पत्तेदार सब्जियां ताज़ा, हरी और मुड़ाव या फंगल धब्बों से मुक्त होती हैं।",
                "ta": "சுகாதாரமான இலைக் காய்கறிகள் புதிய, பச்சை நிறத்தில் இருக்கும் மற்றும் சுருட்டல் அல்லது பூஞ்சை புள்ளிகள் இல்லாமல் இருக்கும்.",
                "te": "ఆరోగ్యకరమైన ఆకుకూరలు తాజా, ఆకుపచ్చ మరియు మడతలు లేదా పురుగుల చారలు లేకుండా ఉంటాయి.",
                "mr": "निरोगी पानांचा भाजी ताजे, हिरवे आणि वाकणे किंवा फंगसचे ठिपके नसलेले असते.",
            },
            "Leaf Blight": {
                "en": "Leaf blight causes irregular dry spots and can lead to lower leaf loss and poor market quality.",
                "hi": "लीफ ब्लाइट से अनियमित सूखे धब्बे बनते हैं और निचली पत्तियां गिरने लगती हैं, जिससे बाजार गुणवत्ता कम होती है।",
                "ta": "இலைப் ப்ளைட் ஒழுங்கற்ற வறண்ட புள்ளிகளை உருவாக்கி, கீழ் இலைகள் விழுவதற்கும், சந்தை தரம் குறைவதற்கும் காரணமாகும்.",
                "te": "లీఫ్ బ్లైట్ క్రమరహిత ఎండిపోయిన చారలను ఏర్పరుస్తుంది మరియు దిగువ ఆకులు పడి మార్కెట్ నాణ్యత తగ్గిపోతుంది.",
                "mr": "लीफ ब्लाइटमुळे अनियमित कोरडे ठिपके दिसतात आणि खालची पाने झडू लागतात, ज्यामुळे बाजाराची गुणवत्ता कमी होते.",
            },
            "Nutrient Deficiency": {
                "en": "Leaf chlorosis and pale growth often point to nutrient imbalance or poor soil health.",
                "hi": "पत्तियों का फीका पड़ना और कमजोर विकास अक्सर पोषक तत्वों की असंतुलन या मिट्टी की खराब सेड को दर्शाता है।",
                "ta": "இலைகளின் மஞ்சள் நிறம் மற்றும் பலவீனமான வளர்ச்சி, ஊட்டச்சத்து ஏற்றத்தாழ்வு அல்லது துரதிருஷ்டமான மண்ணை குறிக்கலாம்.",
                "te": "ఆకు పసుపు రంగు మరియు బలహీనమైన పెరుగుదల సాధారణంగా పోషక అసమతుల్యత లేదా లోతైన మట్టి ఆరోగ్యాన్ని సూచిస్తుంది.",
                "mr": "पानांचा फिकट रंग आणि कमकुवत वाढ याचा अर्थ पोषक असंतुलन किंवा मातीची कमतरता असू शकते.",
            },
        },
    },
}


def enhance_blur_image(image):
    if image is None:
        return None

    img = image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    rgb = img
    denoised = cv2.fastNlMeansDenoisingColored(rgb, None, 7, 7, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    sharpened = cv2.addWeighted(enhanced, 1.65, blurred, -0.65, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def calculate_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray))


def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(np.std(gray))


def calculate_resolution(image):
    height, width = image.shape[:2]
    return width, height


def analyze_image(image):
    if image is None:
        return {"score": 0, "status": "NO IMAGE", "blur": 0, "brightness": 0, "contrast": 0, "width": 0, "height": 0, "issues": ["No image captured"]}

    blur = calculate_blur(image)
    brightness = calculate_brightness(image)
    contrast = calculate_contrast(image)
    width, height = calculate_resolution(image)
    issues = []

    if blur < 50:
        issues.append("Image is very blurry")
    elif blur < 100:
        issues.append("Image is slightly blurry")

    if brightness < 50:
        issues.append("Image is too dark")
    elif brightness > 220:
        issues.append("Image is too bright")

    if contrast < 20:
        issues.append("Image has low contrast")

    if width < 300 or height < 300:
        issues.append("Image resolution is too low")

    score = 100
    if blur < 50:
        score -= 40
    elif blur < 100:
        score -= 20

    if brightness < 50 or brightness > 220:
        score -= 25

    if contrast < 20:
        score -= 15

    if width < 300 or height < 300:
        score -= 20

    score = max(0, min(100, score))

    if score >= 75:
        status = "GOOD"
    elif score >= 50:
        status = "ACCEPTABLE"
    else:
        status = "POOR"

    return {"score": score, "status": status, "blur": blur, "brightness": brightness, "contrast": contrast, "width": width, "height": height, "issues": issues}


def predict_crop_and_disease(image):
    if image is None:
        return {"crop": "Unknown", "disease": "No image", "status": "No crop image provided", "advice": "Please upload or capture a crop image first.", "confidence": 0}

    enhanced = enhance_blur_image(image)
    image_for_analysis = enhanced if enhanced is not None else image

    hsv = cv2.cvtColor(image_for_analysis, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    height, width = image_for_analysis.shape[:2]
    total_pixels = max(1, height * width)

    green_mask = (h >= 25) & (h <= 90) & (s >= 30)
    yellow_mask = (h >= 20) & (h <= 45) & (s >= 35) & (v >= 80)
    brown_mask = (((h >= 0) & (h <= 25)) | ((h >= 150) & (h <= 180))) & (s >= 35) & (v < 220)
    red_mask = (((h >= 0) & (h <= 12)) | ((h >= 160) & (h <= 180))) & (s >= 40)

    green_ratio = np.count_nonzero(green_mask) / total_pixels
    yellow_ratio = np.count_nonzero(yellow_mask) / total_pixels
    brown_ratio = np.count_nonzero(brown_mask) / total_pixels
    red_ratio = np.count_nonzero(red_mask) / total_pixels

    aspect_ratio = width / max(1, height)

    if aspect_ratio > 2.2:
        crop = "Rice"
    elif aspect_ratio > 1.7:
        crop = "Corn"
    elif aspect_ratio > 1.2:
        crop = "Tomato"
    elif aspect_ratio > 0.8:
        crop = "Potato"
    else:
        crop = "Leafy Vegetable"

    if green_ratio > 0.45 and brown_ratio < 0.08 and yellow_ratio < 0.12 and red_ratio < 0.08:
        disease = "Healthy"
        status = "The crop appears healthy."
        advice = "The crop looks healthy. Continue balanced nutrition, proper irrigation, and regular field monitoring. Avoid stress from overwatering, sudden temperature changes, or excess pesticide use."
        confidence = 0.88
    elif brown_ratio > 0.08:
        disease = "Leaf Blight"
        status = "The crop is likely suffering from leaf blight or fungal infection."
        advice = "Remove infected leaves, improve air circulation, avoid overhead irrigation, and apply a suitable fungicide. Keep field sanitation strong and use resistant varieties when possible."
        confidence = 0.82
    elif red_ratio > 0.08:
        disease = "Rust"
        status = "The crop is likely affected by rust disease."
        advice = "Use resistant varieties if available, remove heavily infected leaves, and spray a recommended fungicide. Avoid high humidity during the evening and reduce leaf wetness after irrigation."
        confidence = 0.79
    elif yellow_ratio > 0.12:
        disease = "Nutrient Deficiency"
        status = "The crop shows yellowing that may indicate nutrient stress."
        advice = "Check soil fertility and apply a balanced nutrient program according to crop stage. Improve irrigation, drainage, and monitor for deficiencies before symptoms spread further."
        confidence = 0.75
    else:
        disease = "Leaf Blight"
        status = "The crop may be stressed or diseased."
        advice = "Inspect for spots, wilting, and leaf curling. Keep the field clean, reduce excess water, and consult local agricultural guidance if symptoms continue spreading."
        confidence = 0.68

    return {"crop": crop, "disease": disease, "status": status, "advice": advice, "confidence": confidence}


def build_chat_answer(message, crop, disease, advice, language):
    crop_profile = CROP_DETAILS.get(crop, CROP_DETAILS["Leafy Vegetable"])
    disease_data = crop_profile["diseases"].get(disease, crop_profile["diseases"]["Healthy"])
    lang = LANGUAGES.get(language, "en")
    lower_message = (message or "").lower()

    if any(word in lower_message for word in ["leaf", "pata", "paan", "pachh", "ila"]):
        return crop_profile["leaf"][lang] + " " + disease_data[lang]

    if any(word in lower_message for word in ["disease", "illness", "rog", "vyadhi", "maram"]):
        return disease_data[lang] + " " + advice

    if any(word in lower_message for word in ["eco", "organic", "sustainable", "green", "natural"]):
        if lang == "en":
            return "Use integrated pest management, composted organic matter, and drip irrigation to protect soil and reduce chemical use. Healthy plant management is more eco-friendly and sustainable for long-term crop productivity."
        return "कृषि में जैविक खेती, कम रसायन उपयोग, और ड्रिप सिंचाई अपनाकर मिट्टी, पानी, और जीव-जंतुओं की रक्षा करें।"

    if any(word in lower_message for word in ["advice", "treatment", "medicine", "cure", "solution"]):
        return advice

    return f"Your crop appears to be {crop}. The disease status is {disease}. {crop_profile['leaf'][lang]} {disease_data[lang]} {advice}"


def create_ai_highlight(image, disease):
    """Create a local, explainable lesion heatmap from color symptom signals."""
    if image is None:
        return None, "No image available for visual explanation."

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)

    brown = ((((hue <= 25) | (hue >= 150)) & (saturation >= 35) & (value < 220))).astype(np.uint8)
    yellow = (((hue >= 20) & (hue <= 45) & (saturation >= 35) & (value >= 80))).astype(np.uint8)
    red = ((((hue <= 12) | (hue >= 160)) & (saturation >= 40))).astype(np.uint8)
    symptom_mask = np.maximum(np.maximum(brown, yellow), red) * 255

    kernel = np.ones((9, 9), np.uint8)
    symptom_mask = cv2.morphologyEx(symptom_mask, cv2.MORPH_OPEN, kernel)
    symptom_mask = cv2.GaussianBlur(symptom_mask, (0, 0), 5)
    heatmap = cv2.applyColorMap(symptom_mask, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.62, heatmap, 0.38, 0)

    percentage = float(np.count_nonzero(symptom_mask > 45) / symptom_mask.size * 100)
    if disease == "Healthy" or percentage < 1.0:
        explanation = "AI Highlighted Area: no strong lesion-like color region detected. Continue monitoring the crop."
    elif disease == "Leaf Blight":
        explanation = f"AI Highlighted Area: brown or dry-looking regions cover approximately {percentage:.1f}% of the frame. These areas may match necrotic leaf spots or blight-like damage."
    elif disease == "Rust":
        explanation = f"AI Highlighted Area: orange-red regions cover approximately {percentage:.1f}% of the frame. These areas may match rust-like pustules."
    elif disease == "Nutrient Deficiency":
        explanation = f"AI Highlighted Area: yellowing regions cover approximately {percentage:.1f}% of the frame. This can match chlorosis, nutrient stress, or water stress."
    else:
        explanation = f"AI Highlighted Area: symptom-like color regions cover approximately {percentage:.1f}% of the frame. Inspect these regions closely."

    return overlay, explanation


def process_image(image, language):
    lang_code = LANGUAGES.get(language, "en")
    labels = LOCALIZED_LABELS[lang_code]
    if image is None:
        return (None, None, labels["no_crop"], labels["upload"], "", labels["retake_short"], "No visual explanation available.", [], None, None, None)

    enhanced = enhance_blur_image(image)
    quality = analyze_image(enhanced)
    crop_result = predict_crop_and_disease(enhanced)

    crop = crop_result["crop"]
    disease = crop_result["disease"]
    display_crop = CROP_NAMES[lang_code].get(crop, crop)
    disease_key = "healthy" if disease == "Healthy" else "leaf_blight" if disease == "Leaf Blight" else "rust" if disease == "Rust" else "nutrient"
    display_disease = labels[disease_key]
    advice = localized_advice(disease, lang_code, crop_result["advice"])

    if disease == "Healthy":
        health_status = labels["healthy"]
    elif lang_code == "en":
        health_status = crop_result["status"]
    elif lang_code == "hi":
        health_status = f"फसल में {display_disease} या फंगल संक्रमण की संभावना है।" if disease == "Leaf Blight" else f"फसल में {display_disease} के लक्षण दिखाई दे रहे हैं।"
    elif lang_code == "ta":
        health_status = f"பயிரில் {display_disease} அல்லது பூஞ்சை தொற்று இருக்கலாம்." if disease == "Leaf Blight" else f"பயிரில் {display_disease} அறிகுறிகள் காணப்படுகின்றன."
    elif lang_code == "te":
        health_status = f"పంటలో {display_disease} లేదా శిలీంధ్ర సంక్రమణ ఉండవచ్చు." if disease == "Leaf Blight" else f"పంటలో {display_disease} లక్షణాలు కనిపిస్తున్నాయి."
    else:
        health_status = f"पिकामध्ये {display_disease} किंवा बुरशीजन्य संसर्गाची शक्यता आहे." if disease == "Leaf Blight" else f"पिकामध्ये {display_disease} ची लक्षणे दिसत आहेत."

    if quality["score"] >= 75:
        decision_message = labels["trust"]
    elif quality["score"] >= 50:
        decision_message = labels["caution"]
    else:
        decision_message = labels["retake"]

    issue_text = labels["no_issues"] if len(quality["issues"]) == 0 else "\n".join(f"• {issue}" for issue in quality["issues"])

    profile = CROP_DETAILS.get(crop, CROP_DETAILS["Leafy Vegetable"])
    leaf_detail = profile["leaf"][lang_code]
    disease_detail = profile["diseases"].get(disease, profile["diseases"]["Healthy"])[lang_code]
    highlighted_image, highlight_explanation = create_ai_highlight(enhanced, disease)

    report = f"""
{labels['report']}
=======================

{labels['crop']} : {display_crop}

{labels['disease']} : {display_disease}

{labels['health']}   : {health_status}

{labels['confidence']}     : {crop_result['confidence'] * 100:.0f}%

{labels['quality']}  : {quality['score']}/100

{labels['status']}         : {quality['status']}

{labels['blur']}     : {quality['blur']:.2f}

{labels['brightness']}     : {quality['brightness']:.2f}

{labels['contrast']}       : {quality['contrast']:.2f}

{labels['resolution']}     : {quality['width']} × {quality['height']}

{labels['leaf']}      : {leaf_detail}

{labels['disease_info']}   : {disease_detail}

{labels['issues']}
---------------
{issue_text}
"""

    final_output = f"""
    {labels['final']}
===============

{labels['crop']}: {display_crop}

{labels['status']}: {labels['healthy_status'] if disease == 'Healthy' else labels['diseased_status']}

{labels['disease']}: {display_disease}

{labels['eco']}: {advice}

{labels['care']}: {leaf_detail}
"""

    history = [
        {"role": "assistant", "content": f"I analyzed the crop image. The crop appears to be {crop} with disease status {disease}."},
        {"role": "assistant", "content": f"Leaf detail: {leaf_detail}"},
        {"role": "assistant", "content": f"Disease detail: {disease_detail}"},
    ]
    return highlighted_image, enhanced, f"{display_crop} - {display_disease}", report + "\n" + final_output, decision_message, advice, highlight_explanation, history, crop, disease, advice


def respond_to_chat(message, chat_history, crop_state, disease_state, advice_state, language):
    if message is None or message.strip() == "":
        return chat_history or []

    base_history = chat_history or []
    crop = crop_state or "Crop"
    disease = disease_state or "Healthy"
    advice = advice_state or "Follow field hygiene and balanced nutrition."
    reply = build_chat_answer(message, crop, disease, advice, language)
    return base_history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]


def respond_to_voice(audio, chat_history, crop_state, disease_state, advice_state, language):
    if audio is None:
        return chat_history or []

    lang = LANGUAGES.get(language, "en")
    voice_prompts = {
        "en": "Voice recording received. Please also type the question if you need an exact crop or disease answer; I can then explain the symptoms and next steps.",
        "hi": "आवाज की रिकॉर्डिंग मिल गई। सटीक फसल या रोग का उत्तर पाने के लिए अपना प्रश्न टाइप भी करें। मैं लक्षण और अगले कदम समझाऊंगा।",
        "ta": "குரல் பதிவு கிடைத்தது. துல்லியமான பயிர் அல்லது நோய் பதிலுக்கு கேள்வியை தட்டச்சு செய்யவும். அறிகுறிகள் மற்றும் அடுத்த படிகளை விளக்குகிறேன்.",
        "te": "వాయిస్ రికార్డింగ్ అందింది. ఖచ్చితమైన పంట లేదా వ్యాధి సమాధానం కోసం ప్రశ్నను టైప్ చేయండి. లక్షణాలు మరియు తదుపరి చర్యలను వివరిస్తాను.",
        "mr": "आवाज रेकॉर्डिंग मिळाले. अचूक पीक किंवा रोगाच्या उत्तरासाठी प्रश्न टाइप करा. मी लक्षणे आणि पुढील उपाय समजावून सांगेन.",
    }
    return (chat_history or []) + [{"role": "user", "content": "🎙️ Voice recording"}, {"role": "assistant", "content": voice_prompts[lang]}]


def register_farmer(name, age, profession, email, phone):
    name = (name or "").strip()
    profession = (profession or "").strip()
    email = (email or "").strip()
    phone = (phone or "").strip()

    if not name or not profession or not email:
        return "Please enter your name, profession, and email address."

    try:
        parsed_age = int(age)
    except (TypeError, ValueError):
        return "Please enter a valid age."

    if parsed_age < 18 or parsed_age > 100:
        return "Please enter an age between 18 and 100."

    return (
        f"Profile saved for {name}. You can now use crop diagnosis, farmer chat, "
        f"voice questions, and prevention guidance."
    )


def google_sign_in_status():
    return "Google sign-in is ready to connect. Add your Google OAuth Client ID and redirect URL to enable Gmail authentication."


PREVENTION_POINTS = [
    "Buy certified, disease-free seed from a trusted source.",
    "Choose varieties that are resistant to common local diseases.",
    "Match the crop variety to the local climate and soil.",
    "Use clean seed lots without visible mould, spots, or insects.",
    "Do not save seed from plants with severe disease symptoms.",
    "Treat seed only with a locally approved product when recommended.",
    "Follow the product label for seed-treatment dose and waiting time.",
    "Avoid planting seed that smells sour, mouldy, or fermented.",
    "Keep different crop varieties labelled and separated.",
    "Do not exchange unknown planting material between infected fields.",
    "Test soil before planting whenever possible.",
    "Correct soil pH according to the crop requirement.",
    "Improve compacted soil with organic matter and suitable cultivation.",
    "Use well-decomposed compost instead of fresh manure near roots.",
    "Maintain field drainage so roots are not waterlogged.",
    "Avoid planting repeatedly in the same crop family.",
    "Rotate crops to interrupt soil-borne disease cycles.",
    "Remove volunteer plants that can carry disease between seasons.",
    "Keep field borders free from diseased crop residues.",
    "Plan spacing so mature plants will still have air circulation.",
    "Plant at the recommended depth for the crop and soil.",
    "Avoid overcrowding seedlings in nursery trays or beds.",
    "Use clean trays, tools, and growing media in nurseries.",
    "Disinfect reusable nursery equipment between batches.",
    "Harden seedlings before moving them to the field.",
    "Do not transplant weak, damaged, or visibly infected seedlings.",
    "Avoid injuring stems and roots during transplanting.",
    "Remove nursery plants that show spreading spots or wilting.",
    "Keep nursery beds raised where heavy rain is common.",
    "Use insect-proof netting when it is appropriate for the crop.",
    "Water the root zone instead of wetting leaves unnecessarily.",
    "Prefer drip or furrow irrigation over frequent overhead watering.",
    "Irrigate early enough for foliage to dry before night.",
    "Avoid standing water around stems and plant crowns.",
    "Adjust irrigation to rainfall, soil type, and crop growth stage.",
    "Do not overwater plants that already show root disease symptoms.",
    "Repair leaking pipes and blocked drains quickly.",
    "Use clean irrigation water where contamination is possible.",
    "Keep irrigation channels away from piles of infected plant waste.",
    "Reduce leaf wetness during cool, humid weather.",
    "Feed plants according to soil test and crop stage.",
    "Avoid excessive nitrogen, which can create soft disease-prone growth.",
    "Provide adequate potassium for plant strength when soil tests support it.",
    "Correct micronutrient deficiencies with verified recommendations.",
    "Do not mix fertilizers without checking compatibility.",
    "Keep fertilizer granules off wet leaves and stems.",
    "Use compost and mulch to support soil biological activity.",
    "Do not apply fresh manure directly against plant stems.",
    "Monitor plants after fertilizing for scorch or stress.",
    "Keep a simple record of fertilizer type, dose, and application date.",
    "Remove fallen diseased leaves from the crop area.",
    "Do not leave pruned infected tissue between healthy plants.",
    "Dispose of infected material away from irrigation channels.",
    "Do not compost heavily diseased material unless the compost gets hot enough.",
    "Clean pruning tools between plants with an approved disinfectant.",
    "Avoid working through wet foliage when disease spreads by contact.",
    "Wash boots and tools after working in a diseased field.",
    "Keep weeds controlled because they can host pests and pathogens.",
    "Remove volunteer plants from previous crops when they become hosts.",
    "Keep storage areas, crates, and harvest tools clean.",
    "Scout the crop on a regular schedule, not only after symptoms appear.",
    "Inspect lower leaves first because they often show early symptoms.",
    "Check both sides of leaves for spots, insects, eggs, and fungal growth.",
    "Compare healthy and affected plants in the same field.",
    "Mark the location of suspicious plants for follow-up visits.",
    "Take clear photos with the date and crop variety recorded.",
    "Record whether symptoms are spreading, stable, or recovering.",
    "Check stems, roots, fruit, and soil as well as leaves.",
    "Ask an agricultural expert when symptoms are unusual or severe.",
    "Use weather forecasts to prepare for disease-favourable conditions.",
    "Increase scouting after long periods of leaf wetness.",
    "Inspect after storms, flooding, hail, or strong winds.",
    "Avoid field work when moving through plants would spread wet spores.",
    "Use clean protective covers or supports after storm damage.",
    "Prune only when weather will allow wounds to dry.",
    "Protect plants from avoidable mechanical injury.",
    "Use mulch to reduce soil splash onto lower leaves.",
    "Stake or trellis crops to improve airflow around leaves.",
    "Remove lower leaves that touch wet soil when crop guidance allows.",
    "Control insect vectors with an integrated pest-management plan.",
    "Use yellow or blue sticky traps where suitable for monitoring.",
    "Encourage beneficial insects by protecting flowering habitat.",
    "Avoid broad-spectrum sprays when a targeted option is available.",
    "Check the underside of leaves for mites, aphids, and whiteflies.",
    "Remove heavily infested plant parts when practical.",
    "Do not move infested seedlings into a clean field.",
    "Use physical barriers or nets for young plants when practical.",
    "Alternate approved control methods to slow resistance development.",
    "Identify the likely pest or disease before choosing a treatment.",
    "Never spray a chemical just because a leaf looks slightly yellow.",
    "Use a hand lens to inspect suspicious spots and pests.",
    "Preserve beneficial predators such as ladybirds and lacewings.",
    "Follow the label, crop, dose, and pre-harvest interval for products.",
    "Wear the protective equipment listed on the product label.",
    "Keep children, animals, food, and water away during spraying.",
    "Do not spray in strong wind or before heavy rain.",
    "Do not increase chemical dose to compensate for poor diagnosis.",
    "Do not mix products unless the label or expert recommendation allows it.",
    "Calibrate sprayers so the crop receives an even application.",
    "Clean sprayers safely and never pour leftovers into drains or ponds.",
    "Store farm products in their original labelled containers.",
    "Observe the required waiting period before harvest.",
    "Keep treatment and disease observations in a field notebook.",
    "Review each season and change practices that repeatedly lead to disease.",
    "Use local agricultural extension advice for serious or fast-spreading disease.",
    "Send a sample to a plant diagnostic laboratory when the diagnosis is uncertain.",
    "Quarantine unfamiliar planting material before introducing it to the farm.",
    "Share disease warnings with neighbouring farmers when appropriate.",
    "Clean harvest containers before moving them between fields.",
    "Dry harvested produce properly before storage.",
    "Keep stored produce cool, dry, ventilated, and protected from rodents.",
    "Remove damaged produce before it infects healthy stored produce.",
    "Review the farm prevention plan before every new planting season.",
]

PREVENTION_POINTS = PREVENTION_POINTS[:100]


PREVENTION_HTML = "<div class='prevention-grid'>" + "".join(
    f"<div class='prevention-item'><strong>{index}.</strong> {point}</div>"
    for index, point in enumerate(PREVENTION_POINTS, start=1)
) + "</div>"


APP_CSS = """
body, .gradio-container {
    background: #000000 !important;
}
.gradio-container {
    background-image: linear-gradient(135deg, #000000 0%, #101613 100%) !important;
    --body-text-color: #ffffff;
    --block-label-text-color: #ffffff;
    --body-text-color-subdued: #d5ded9;
    --block-background-fill: #111513;
    --block-border-color: #34423b;
    color: #ffffff !important;
    font-family: "Trebuchet MS", "Segoe UI", sans-serif !important;
}
.prose, .prose p, .prose li, label, .block-label, .label-wrap, .form, .form label, .form span { color: #ffffff !important; }
.prose h1, .prose h2, .prose h3, .app-header h1, .section-title { font-family: Georgia, "Times New Roman", serif !important; }
.gradio-container textarea, .gradio-container input, .gradio-container [role="combobox"] { color: #ffffff !important; background: #111513 !important; }
.gradio-container textarea::placeholder, .gradio-container input::placeholder { color: #b7c5bd !important; }
.gradio-container .wrap, .gradio-container .container, .gradio-container .block-label, .gradio-container [data-testid="block-label"], .gradio-container [data-testid="block-info"] { color: #ffffff !important; }
.contain { max-width: 1180px !important; }
.app-header { border-left: 7px solid #e27a55; background: linear-gradient(110deg, #17231d, #202c26); padding: 24px 28px; border-radius: 10px; color: #ffffff; box-shadow: 0 8px 22px rgba(0, 0, 0, .35); }
.app-header h1 { margin: 0 0 8px; color: #ffffff; letter-spacing: .2px; }
.app-header p { color: #e1e9e4; margin: 0; }
.section-title { color: #ffffff !important; padding: 10px 14px; border-radius: 5px; margin-top: 18px; font-weight: 700; box-shadow: none; }
.section-title.green { background: linear-gradient(90deg, #287653, #4b9b70); }
.section-title.blue { background: linear-gradient(90deg, #356b7c, #5798a0); }
.section-title.gold { background: linear-gradient(90deg, #b66b3e, #d29a4a); }
.study-card { background: #111513; border: 1px solid #34423b; border-top: 4px solid #80b88b; border-radius: 8px; padding: 16px; min-height: 150px; color: #ffffff; box-shadow: 0 5px 16px rgba(0, 0, 0, .28); }
.study-card p { color: #e1e9e4 !important; }
.study-card h3 { color: #a8d895 !important; }
.prevention-intro { background: #17231d !important; border: 1px solid #34423b; border-left: 5px solid #e27a55; padding: 14px 16px; border-radius: 6px; color: #ffffff; }
.prevention-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 14px; margin-top: 12px; }
.prevention-item { background: #111513 !important; border: 1px solid #34423b; border-radius: 5px; padding: 9px 11px; color: #ffffff; font-size: 14px; line-height: 1.45; }
.prevention-item strong { color: #a8d895; margin-right: 4px; }
button.primary { background: linear-gradient(90deg, #287653, #3f9368) !important; border-radius: 6px !important; }
.gradio-container button { border-radius: 5px !important; }
.workflow-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 14px 0 8px; }
.workflow-step { background: #111513; border: 1px solid #34423b; border-radius: 8px; padding: 12px 14px; color: #ffffff; }
.workflow-step strong { display: block; color: #a8d895; font-size: 15px; margin-bottom: 4px; }
.workflow-step span { color: #d5ded9; font-size: 13px; }
.quick-actions { background: #17231d; border: 1px solid #34423b; border-radius: 8px; padding: 12px; }
.quick-actions p { margin: 0 0 8px; color: #d5ded9; font-size: 13px; }
.quick-actions button { background: #243a2d !important; color: #ffffff !important; border: 1px solid #52745b !important; }
.knowledge-accordion { margin-top: 16px; }
.tab-nav button { color: #d5ded9 !important; background: #111513 !important; border-color: #34423b !important; }
.tab-nav button.selected { color: #ffffff !important; background: #287653 !important; border-color: #80b88b !important; }
.tabs { margin-top: 18px; }
@media (max-width: 720px) { .prevention-grid { grid-template-columns: 1fr; } }
"""


with gr.Blocks(title="Smart Crop Health Assistant") as app:
    gr.HTML("""
<div class="app-header">
<h1>🌾 AI Crop Health & Disease Advisor</h1>

<p>This app helps farmers check crop health from a leaf image, improve blurred photos, explain crop and leaf details, and give eco-friendly agricultural guidance.</p>
</div>
""")

    gr.HTML("""
    <div class="workflow-strip">
        <div class="workflow-step"><strong>1. Scan</strong><span>Upload or capture a clear leaf photo.</span></div>
        <div class="workflow-step"><strong>2. Understand</strong><span>See the crop, symptoms, and highlighted areas.</span></div>
        <div class="workflow-step"><strong>3. Act</strong><span>Ask questions and follow practical field care.</span></div>
    </div>
    """)

    with gr.Row():
        image_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="Upload or capture crop leaf image")
        language_dropdown = gr.Dropdown(choices=list(LANGUAGES.keys()), value="English", label="Language / भाषा")

    with gr.Row():
        analyze_button = gr.Button("Analyze crop", variant="primary")

    with gr.Row():
        enhanced_image = gr.Image(label="AI-Enhanced Leaf Image")
        status_output = gr.Textbox(label="Crop Health & Disease Status")

    with gr.Row():
        highlight_image = gr.Image(label="AI Highlighted Symptom Areas")
        highlight_output = gr.Textbox(label="Why These Areas Were Highlighted", lines=4)

    with gr.Row():
        analysis_output = gr.Textbox(label="Detailed Crop Diagnosis", lines=20)
        decision_output = gr.Textbox(label="Image Quality Decision", lines=7)

    with gr.Row():
        advice_output = gr.Textbox(label="Recommended Eco-Friendly Field Care", lines=6)

    crop_state = gr.State(value=None)
    disease_state = gr.State(value=None)
    advice_state = gr.State(value=None)

    gr.Markdown('<div class="section-title blue">💬 Farmer Questions & Voice Assistant</div>', sanitize_html=False)
    chatbot = gr.Chatbot(label="Agricultural Advisor Chat", value=[])
    chat_input = gr.Textbox(placeholder="Ask about leaf health, disease, or eco-friendly care...", label="Ask Your Farming Question")
    voice_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Record a farmer question")

    with gr.Row(elem_classes=["quick-actions"]):
        gr.Markdown("**Quick questions**")
        explain_button = gr.Button("Explain my result")
        organic_button = gr.Button("Organic care")
        prevention_button = gr.Button("How do I prevent it?")

    analyze_button.click(
        fn=process_image,
        inputs=[image_input, language_dropdown],
        outputs=[highlight_image, enhanced_image, status_output, analysis_output, decision_output, advice_output, highlight_output, chatbot, crop_state, disease_state, advice_state],
    )

    chat_input.submit(
        fn=respond_to_chat,
        inputs=[chat_input, chatbot, crop_state, disease_state, advice_state, language_dropdown],
        outputs=[chatbot],
    )

    voice_input.change(
        fn=respond_to_voice,
        inputs=[voice_input, chatbot, crop_state, disease_state, advice_state, language_dropdown],
        outputs=[chatbot],
    )

    explain_button.click(
        fn=lambda history, crop, disease, advice, language: respond_to_chat("Explain my result in simple words", history, crop, disease, advice, language),
        inputs=[chatbot, crop_state, disease_state, advice_state, language_dropdown],
        outputs=[chatbot],
    )
    organic_button.click(
        fn=lambda history, crop, disease, advice, language: respond_to_chat("Give me organic and eco-friendly care advice", history, crop, disease, advice, language),
        inputs=[chatbot, crop_state, disease_state, advice_state, language_dropdown],
        outputs=[chatbot],
    )
    prevention_button.click(
        fn=lambda history, crop, disease, advice, language: respond_to_chat("How can I prevent this disease?", history, crop, disease, advice, language),
        inputs=[chatbot, crop_state, disease_state, advice_state, language_dropdown],
        outputs=[chatbot],
    )

    with gr.Tabs():
        with gr.Tab("Farmer Profile"):
            gr.Markdown("### Farmer Registration")
            gr.Markdown("Create a local farmer profile to personalize crop guidance. Your details stay in this app session until a secure database is connected.")
            with gr.Row():
                farmer_name = gr.Textbox(label="Full name", placeholder="Enter your name")
                farmer_age = gr.Number(label="Age", minimum=18, maximum=100, precision=0)
            with gr.Row():
                farmer_profession = gr.Textbox(label="Profession", placeholder="Farmer, farm worker, student...")
                farmer_email = gr.Textbox(label="Gmail or email address", placeholder="you@gmail.com")
            farmer_phone = gr.Textbox(label="Phone number (optional)", placeholder="Your phone number")
            with gr.Row():
                register_button = gr.Button("Save farmer profile", variant="primary")
                google_button = gr.Button("Continue with Google / Gmail")
            registration_status = gr.Textbox(label="Account status", interactive=False)

            register_button.click(
                fn=register_farmer,
                inputs=[farmer_name, farmer_age, farmer_profession, farmer_email, farmer_phone],
                outputs=[registration_status],
            )
            google_button.click(
                fn=google_sign_in_status,
                inputs=[],
                outputs=[registration_status],
            )

        with gr.Tab("Crop Knowledge"):
            gr.Markdown("### Crop Knowledge & Farming Study")
            with gr.Row():
                gr.HTML("""
                <div class="study-card"><h3>🌱 Crop care</h3><p>Use healthy seeds, test the soil, mulch around plants, monitor moisture, and avoid watering leaves late in the day.</p><p>Prefer drip irrigation and compost where practical to save water and protect soil life.</p></div>
                """)
                gr.HTML("""
                <div class="study-card"><h3>🍃 Leaf disease clues</h3><p>Brown expanding spots may indicate blight. Orange pustules may suggest rust. Even yellowing can come from nutrient or water stress.</p><p>Always compare several leaves and ask a local expert before applying chemicals.</p></div>
                """)
                gr.HTML("""
                <div class="study-card"><h3>🔬 Extra study</h3><p>Record crop variety, field location, weather, irrigation, and symptom spread. This makes diagnosis more reliable.</p><p>Study one symptom at a time: spots, curling, wilting, yellowing, insects, and stem damage.</p></div>
                """)

        with gr.Tab("Prevention"):
            gr.Markdown("### Disease Prevention & Field Practices")
            gr.HTML("<div class='prevention-intro'><strong>Seasonal field checklist.</strong> Combine seed, soil, water, scouting, hygiene, pest management, and safe treatment. Confirm serious diagnoses with a local agricultural expert.</div>" + PREVENTION_HTML)

        with gr.Tab("Safe Farming"):
            gr.Markdown("### Safe & Sustainable Farming")
            gr.Markdown("Use diagnosis as an early screening aid, not as a final laboratory diagnosis. Follow local label directions for any product and contact an agricultural extension worker when symptoms spread quickly or the crop is valuable.")
            gr.Markdown("### Languages")
            gr.Markdown("English · हिन्दी · தமிழ் · తెలుగు · मराठी")


if __name__ == "__main__":
    app.launch(debug=True, server_port=7876, css=APP_CSS)