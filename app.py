"""
Flask Web Application - Disease Prediction System
Hỗ trợ tiếng Việt
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from disease_predictor import DiseasePredictionModel
from translations import get_symptom_vi, get_disease_vi, SYMPTOMS_VI, DISEASES_VI
import os
import sys
import logging
from datetime import datetime
os.makedirs('logs', exist_ok=True)
# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Khởi tạo model
predictor = DiseasePredictionModel()

# Load model khi khởi động
import subprocess

if not os.path.exists("models/disease_model.pkl"):
    print("📥 Model not found → downloading...")
    subprocess.run([sys.executable, "download_model.py"], check=True)

try:
    predictor.load_model('models')
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.warning(f"⚠️  Model not loaded: {e}")
    logger.info("💡 Please train the model first using: python train.py <csv_file>")


@app.route('/')
def home():
    """Trang chủ"""
    # Tạo danh sách triệu chứng với cả tiếng Anh và tiếng Việt
    symptoms_with_vi = []
    for symptom_en in predictor.symptoms_list:
        symptoms_with_vi.append({
            'en': symptom_en,
            'vi': get_symptom_vi(symptom_en)
        })
    
    return render_template('index.html', 
                         symptoms=symptoms_with_vi,
                         n_symptoms=len(predictor.symptoms_list),
                         n_diseases=len(predictor.diseases_list))


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint để dự đoán bệnh"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        selected_symptoms = data.get('symptoms', [])
        
        if not selected_symptoms:
            return jsonify({
                'success': False,
                'error': 'Vui lòng chọn ít nhất một triệu chứng'
            }), 400
        
        # Log request
        logger.info(f"Prediction request with {len(selected_symptoms)} symptoms")
        
        # Dự đoán
        results = predictor.predict_disease(selected_symptoms)
        
        # Dịch kết quả sang tiếng Việt
        results_vi = {
            'primary_prediction': results['primary_prediction'],
            'primary_prediction_vi': get_disease_vi(results['primary_prediction']),
            'confidence': results['confidence'],
            'matched_symptoms': results['matched_symptoms'],
            'matched_symptoms_vi': [get_symptom_vi(s) for s in results['matched_symptoms']],
            'unmatched_symptoms': results['unmatched_symptoms'],
            'total_symptoms_checked': results['total_symptoms_checked'],
            'top_predictions': []
        }
        
        # Dịch top predictions
        for pred in results['top_predictions']:
            results_vi['top_predictions'].append({
                'disease': pred['disease'],
                'disease_vi': get_disease_vi(pred['disease']),
                'probability': pred['probability'],
                'percentage': pred['percentage']
            })
        
        # Log result
        logger.info(f"Predicted: {results['primary_prediction']} ({results['confidence']*100:.1f}%)")
        
        return jsonify({
            'success': True,
            'results': results_vi,
            'timestamp': datetime.now().isoformat()
        })
    
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Đã xảy ra lỗi khi dự đoán. Vui lòng thử lại.'
        }), 500


@app.route('/api/symptoms')
def get_symptoms():
    """API để lấy danh sách tất cả triệu chứng (cả tiếng Anh và tiếng Việt)"""
    try:
        symptoms_list = []
        for symptom_en in predictor.symptoms_list:
            symptoms_list.append({
                'en': symptom_en,
                'vi': get_symptom_vi(symptom_en)
            })
        
        return jsonify({
            'success': True,
            'symptoms': symptoms_list,
            'total': len(symptoms_list)
        })
    except Exception as e:
        logger.error(f"Error getting symptoms: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/diseases')
def get_diseases():
    """API để lấy danh sách tất cả bệnh (cả tiếng Anh và tiếng Việt)"""
    try:
        diseases_list = []
        for disease_en in predictor.diseases_list:
            diseases_list.append({
                'en': disease_en,
                'vi': get_disease_vi(disease_en)
            })
        
        return jsonify({
            'success': True,
            'diseases': diseases_list,
            'total': len(diseases_list)
        })
    except Exception as e:
        logger.error(f"Error getting diseases: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/disease/<disease_name>')
def get_disease_info(disease_name):
    """API để lấy thông tin về một bệnh cụ thể"""
    try:
        symptoms = predictor.get_disease_symptoms(disease_name)
        
        symptoms_vi = [get_symptom_vi(s) for s in symptoms]
        
        return jsonify({
            'success': True,
            'disease': disease_name,
            'disease_vi': get_disease_vi(disease_name),
            'symptoms': symptoms,
            'symptoms_vi': symptoms_vi,
            'total_symptoms': len(symptoms)
        })
    except Exception as e:
        logger.error(f"Error getting disease info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'n_symptoms': len(predictor.symptoms_list),
        'n_diseases': len(predictor.diseases_list),
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Tạo thư mục logs nếu chưa có

    
    # Hiển thị thông tin
    print("=" * 70)
    print(" " * 20 + "DISEASE PREDICTION SYSTEM")
    print("=" * 70)
    print(f"\n🏥 Hệ thống dự đoán bệnh đã sẵn sàng!")
    print(f"📊 Model status: {'✅ Loaded' if predictor.model else '❌ Not loaded'}")
    print(f"💊 Số bệnh: {len(predictor.diseases_list)}")
    print(f"🩺 Số triệu chứng: {len(predictor.symptoms_list)}")
    print(f"\n🌐 Server đang chạy tại: http://localhost:5002")
    print(f"📖 API Documentation:")
    print(f"   - GET  /api/health       - Health check")
    print(f"   - GET  /api/symptoms     - Lấy danh sách triệu chứng")
    print(f"   - GET  /api/diseases     - Lấy danh sách bệnh")
    print(f"   - POST /api/predict      - Dự đoán bệnh")
    print(f"   - GET  /api/disease/<n>  - Thông tin bệnh")
    print("=" * 70)
    print("\n⏳ Starting server...\n")
    
    # Chạy server
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False)
