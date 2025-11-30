"""
Disease Prediction Model
Hệ thống dự đoán bệnh dựa trên triệu chứng
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import pickle
import json
import os
from datetime import datetime


class DiseasePredictionModel:
    """Class chính để xử lý dự đoán bệnh"""
    
    def __init__(self):
        self.model = None
        self.symptoms_list = []
        self.diseases_list = []
        self.df = None
        self.training_history = []
        
    def load_data(self, filepath):
        """
        Load dataset từ file CSV
        Cấu trúc: Cột đầu tiên = tên bệnh, các cột sau = triệu chứng (0/1)
        """
        print(f"📂 Đang load dữ liệu từ: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
        
        # Load CSV
        self.df = pd.read_csv(filepath)
        
        # Lấy danh sách bệnh và triệu chứng
        self.diseases_list = self.df.iloc[:, 0].unique().tolist()
        self.symptoms_list = self.df.columns[1:].tolist()
        
        print(f"✅ Đã load thành công!")
        print(f"   - Số bệnh: {len(self.diseases_list)}")
        print(f"   - Số triệu chứng: {len(self.symptoms_list)}")
        print(f"   - Tổng số mẫu: {len(self.df)}")
        
        return self.df
    
    def prepare_data(self):
        """Chuẩn bị dữ liệu cho training"""
        if self.df is None:
            raise Exception("Chưa load dữ liệu! Hãy gọi load_data() trước.")
        
        print("\n🔧 Đang chuẩn bị dữ liệu...")
        
        # X: features (triệu chứng), y: labels (tên bệnh)
        X = self.df.iloc[:, 1:].values
        y = self.df.iloc[:, 0].values
        
        print(f"✅ Dữ liệu đã sẵn sàng!")
        print(f"   - Shape X: {X.shape}")
        print(f"   - Shape y: {y.shape}")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train model - Tối ưu cho dataset lớn"""
        print("\n🚀 Bắt đầu training model...")
        print(f"   - Tỷ lệ train/test: {int((1-test_size)*100)}/{int(test_size*100)}")
        
        # Kiểm tra phân bố class
        class_counts = Counter(y)
        min_samples = min(class_counts.values())
        
        print(f"\n📊 Phân tích dữ liệu:")
        print(f"   - Tổng số class (bệnh): {len(class_counts)}")
        print(f"   - Số mẫu ít nhất: {min_samples}")
        print(f"   - Số mẫu nhiều nhất: {max(class_counts.values())}")
        
        # Lọc bỏ class có ít hơn 2 mẫu
        if min_samples < 2:
            print(f"\n⚠️  Cảnh báo: Có {sum(1 for c in class_counts.values() if c < 2)} bệnh chỉ có 1 mẫu")
            print("   → Đang lọc bỏ các bệnh có quá ít mẫu...")
            
            valid_indices = []
            for i, label in enumerate(y):
                if class_counts[label] >= 2:
                    valid_indices.append(i)
            
            X = X[valid_indices]
            y = y[valid_indices]
            
            print(f"   ✓ Đã lọc: {len(X)} mẫu còn lại")
            self.diseases_list = list(set(y))
            class_counts = Counter(y)
        
        # SAMPLING để giảm dataset
        MAX_SAMPLES_PER_CLASS = 200  # Giới hạn mỗi bệnh tối đa 200 mẫu
        
        if max(class_counts.values()) > MAX_SAMPLES_PER_CLASS:
            print(f"\n🔧 Dataset quá lớn! Đang sampling {MAX_SAMPLES_PER_CLASS} mẫu/bệnh...")
            
            sampled_indices = []
            for disease in set(y):
                disease_indices = np.where(y == disease)[0]
                if len(disease_indices) > MAX_SAMPLES_PER_CLASS:
                    sampled = np.random.choice(disease_indices, MAX_SAMPLES_PER_CLASS, replace=False)
                    sampled_indices.extend(sampled)
                else:
                    sampled_indices.extend(disease_indices)
            
            X = X[sampled_indices]
            y = y[sampled_indices]
            
            print(f"   ✓ Giảm xuống: {len(X)} mẫu ({len(set(y))} bệnh)")
        
        # ENCODE LABELS
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n🔧 Đã encode {len(self.label_encoder.classes_)} bệnh thành số 0-{len(self.label_encoder.classes_)-1}")
        
        # Chia dữ liệu
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded,
                test_size=test_size, 
                random_state=random_state,
                stratify=y_encoded
            )
            
            print(f"\n✅ Đã chia dữ liệu:")
            print(f"   - Số mẫu train: {len(X_train)}")
            print(f"   - Số mẫu test: {len(X_test)}")
            
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=test_size, 
                random_state=random_state,
                stratify=None
            )
        
        # Khởi tạo XGBoost - TỐI ƯU
        print("\n⚙️  Cấu hình XGBoost model (tối ưu):")
        from xgboost import XGBClassifier
        
        self.model = XGBClassifier(
            n_estimators=100,          # Giảm xuống 100
            max_depth=8,               # Giảm độ sâu
            learning_rate=0.1,
            subsample=0.8,             # Chỉ dùng 80% dữ liệu mỗi tree
            colsample_bytree=0.8,      # Chỉ dùng 80% features mỗi tree
            n_jobs=4,                  # Giảm xuống 4 cores
            random_state=random_state,
            tree_method='hist',        # Nhanh nhất
            verbosity=1,
            eval_metric='mlogloss'
        )
        
        print("   - Estimators: 100")
        print("   - Max depth: 8")
        print("   - Subsample: 0.8")
        print("   - Parallel jobs: 4")
        
        # Train với progress tracking
        print("\n⏳ Đang training (ước tính 2-5 phút)...")
        import time
        start_time = time.time()
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=True
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ Training hoàn tất! (Thời gian: {training_time:.1f}s)")
        
        # Đánh giá
        print("\n📊 Đang đánh giá model...")
        
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        print("\n📊 KẾT QUẢ ĐÁNH GIÁ:")
        print("=" * 50)
        print(f"Độ chính xác trên tập TRAIN: {train_accuracy*100:.2f}%")
        print(f"Độ chính xác trên tập TEST:  {test_accuracy*100:.2f}%")
        print("=" * 50)
        
        # Metrics
        y_pred = self.model.predict(X_test)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        print("\n📋 Metrics Summary:")
        print(f"   - Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"   - Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"   - Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"   - F1-Score:  {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        
        # Lưu lịch sử
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'n_samples': len(X),
            'test_size': test_size,
            'training_time': training_time
        })
        
        return self.model
    
    def predict_disease(self, selected_symptoms):
            """
            Dự đoán bệnh dựa trên triệu chứng
            """
            if self.model is None:
                raise Exception("Model chưa được train! Hãy train model trước.")
            
            if not isinstance(selected_symptoms, list):
                raise TypeError("selected_symptoms phải là list")
            
            if len(selected_symptoms) == 0:
                raise ValueError("Vui lòng chọn ít nhất 1 triệu chứng")
            
            # Tạo vector triệu chứng
            symptom_vector = np.zeros(len(self.symptoms_list))
            
            matched_symptoms = []
            unmatched_symptoms = []
            
            for symptom in selected_symptoms:
                if symptom in self.symptoms_list:
                    idx = self.symptoms_list.index(symptom)
                    symptom_vector[idx] = 1
                    matched_symptoms.append(symptom)
                else:
                    unmatched_symptoms.append(symptom)
            
            # Reshape cho predict
            symptom_vector = symptom_vector.reshape(1, -1)
            
            # Dự đoán (encoded)
            predicted_encoded = self.model.predict(symptom_vector)[0]
            
            # Decode về tên bệnh
            if hasattr(self, 'label_encoder'):
                predicted_disease = self.label_encoder.inverse_transform([predicted_encoded])[0]
                probabilities = self.model.predict_proba(symptom_vector)[0]
                
                # Top 5
                top_indices = np.argsort(probabilities)[-5:][::-1]
                
                results = {
                    'primary_prediction': predicted_disease,
                    'confidence': float(probabilities[predicted_encoded]),
                    'matched_symptoms': matched_symptoms,
                    'unmatched_symptoms': unmatched_symptoms,
                    'total_symptoms_checked': len(selected_symptoms),
                    'top_predictions': []
                }
                
                for idx in top_indices:
                    disease_name = self.label_encoder.inverse_transform([idx])[0]
                    probability = probabilities[idx]
                    results['top_predictions'].append({
                        'disease': disease_name,
                        'probability': float(probability),
                        'percentage': f"{float(probability)*100:.1f}%"
                    })
            else:
                # Fallback cho Random Forest (không cần encode)
                predicted_disease = self.model.predict(symptom_vector)[0]
                probabilities = self.model.predict_proba(symptom_vector)[0]
                
                top_indices = np.argsort(probabilities)[-5:][::-1]
                
                results = {
                    'primary_prediction': predicted_disease,
                    'confidence': float(probabilities[self.model.classes_.tolist().index(predicted_disease)]),
                    'matched_symptoms': matched_symptoms,
                    'unmatched_symptoms': unmatched_symptoms,
                    'total_symptoms_checked': len(selected_symptoms),
                    'top_predictions': []
                }
                
                for idx in top_indices:
                    disease_name = self.model.classes_[idx]
                    probability = probabilities[idx]
                    results['top_predictions'].append({
                        'disease': disease_name,
                        'probability': float(probability),
                        'percentage': f"{float(probability)*100:.1f}%"
                    })
            
            return results
    
    def get_disease_symptoms(self, disease_name):
        """Lấy danh sách triệu chứng của một bệnh"""
        if self.df is None:
            raise Exception("Chưa load dữ liệu!")
        
        disease_data = self.df[self.df.iloc[:, 0] == disease_name]
        
        if len(disease_data) == 0:
            return []
        
        # Lấy các triệu chứng có giá trị = 1
        symptoms = []
        for col in self.df.columns[1:]:
            if disease_data[col].values[0] == 1:
                symptoms.append(col)
        
        return symptoms
    
    def save_model(self, model_dir='models'):
        """Lưu model và metadata"""
        print(f"\n💾 Đang lưu model vào thư mục: {model_dir}")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Lưu model
        model_path = os.path.join(model_dir, 'disease_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"   ✓ Đã lưu model: {model_path}")
        
        # Lưu label encoder (nếu có)
        if hasattr(self, 'label_encoder'):
            encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"   ✓ Đã lưu label encoder: {encoder_path}")
        
        # Lưu danh sách triệu chứng
        symptoms_path = os.path.join(model_dir, 'symptoms_list.json')
        with open(symptoms_path, 'w', encoding='utf-8') as f:
            json.dump(self.symptoms_list, f, ensure_ascii=False, indent=2)
        print(f"   ✓ Đã lưu danh sách triệu chứng: {symptoms_path}")
        
        # Lưu danh sách bệnh
        diseases_path = os.path.join(model_dir, 'diseases_list.json')
        with open(diseases_path, 'w', encoding='utf-8') as f:
            json.dump(self.diseases_list, f, ensure_ascii=False, indent=2)
        print(f"   ✓ Đã lưu danh sách bệnh: {diseases_path}")
        
        # Lưu metadata
        metadata = {
            'model_type': 'XGBoost' if 'XGB' in str(type(self.model)) else 'RandomForest',
            'n_symptoms': len(self.symptoms_list),
            'n_diseases': len(self.diseases_list),
            'training_history': self.training_history,
            'last_trained': datetime.now().isoformat(),
            'has_label_encoder': hasattr(self, 'label_encoder')
        }
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"   ✓ Đã lưu metadata: {metadata_path}")
        
        print("\n✅ Lưu model thành công!")
    
    def load_model(self, model_dir='models'):
        """Load model đã train"""
        print(f"\n📂 Đang load model từ: {model_dir}")
        
        try:
            # Load model
            model_path = os.path.join(model_dir, 'disease_model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"   ✓ Đã load model")
            
            # Load label encoder (nếu có)
            encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"   ✓ Đã load label encoder")
            
            # Load symptoms
            symptoms_path = os.path.join(model_dir, 'symptoms_list.json')
            with open(symptoms_path, 'r', encoding='utf-8') as f:
                self.symptoms_list = json.load(f)
            print(f"   ✓ Đã load {len(self.symptoms_list)} triệu chứng")
            
            # Load diseases
            diseases_path = os.path.join(model_dir, 'diseases_list.json')
            with open(diseases_path, 'r', encoding='utf-8') as f:
                self.diseases_list = json.load(f)
            print(f"   ✓ Đã load {len(self.diseases_list)} bệnh")
            
            # Load metadata (optional)
            try:
                metadata_path = os.path.join(model_dir, 'metadata.json')
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.training_history = metadata.get('training_history', [])
                print(f"   ✓ Đã load metadata")
            except:
                pass
            
            print("\n✅ Load model thành công!")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Không tìm thấy file model. Vui lòng train model trước.\nLỗi: {e}")
        except Exception as e:
            raise Exception(f"Lỗi khi load model: {e}")


if __name__ == "__main__":
    print("Disease Prediction Model Module")
    print("Sử dụng class DiseasePredictionModel để train và predict")