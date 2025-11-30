"""
Script để train model dự đoán bệnh
Sử dụng: python train.py <path_to_csv_file>
"""

import sys
import os
from disease_predictor import DiseasePredictionModel


def main():
    print("=" * 70)
    print(" " * 15 + "HỆ THỐNG TRAIN MODEL DỰ ĐOÁN BỆNH")
    print("=" * 70)
    
    # Lấy đường dẫn file CSV
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = input("\n📁 Nhập đường dẫn đến file CSV: ").strip()
    
    if not os.path.exists(csv_file):
        print(f"\n❌ Lỗi: Không tìm thấy file '{csv_file}'")
        return
    
    print(f"\n📊 File CSV: {csv_file}")
    
    try:
        # Khởi tạo model
        predictor = DiseasePredictionModel()
        
        # Load dữ liệu
        predictor.load_data(csv_file)
        
        # Hiển thị mẫu dữ liệu
        print("\n" + "=" * 70)
        print("PREVIEW DỮ LIỆU (5 dòng đầu)")
        print("=" * 70)
        print(predictor.df.head())
        
        # Confirm training
        print("\n" + "=" * 70)
        confirm = input("\n❓ Bạn có muốn bắt đầu training không? (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("❌ Đã hủy training.")
            return
        
        # Prepare data
        X, y = predictor.prepare_data()
        
        # Train model
        predictor.train_model(X, y, test_size=0.2)
        
        # Save model
        print("\n" + "=" * 70)
        save = input("\n❓ Bạn có muốn lưu model không? (y/n): ").strip().lower()
        
        if save == 'y':
            model_dir = input("📁 Nhập thư mục lưu model (Enter = 'models'): ").strip()
            if not model_dir:
                model_dir = 'models'
            
            predictor.save_model(model_dir=model_dir)
        
        # Test prediction
        print("\n" + "=" * 70)
        print("TEST DỰ ĐOÁN")
        print("=" * 70)
        test = input("\n❓ Bạn có muốn thử dự đoán không? (y/n): ").strip().lower()
        
        if test == 'y':
            print(f"\nDanh sách một số triệu chứng:")
            for i, symptom in enumerate(predictor.symptoms_list[:10], 1):
                print(f"  {i}. {symptom}")
            print("  ...")
            
            print("\n💡 Nhập các triệu chứng, cách nhau bởi dấu phẩy")
            symptoms_input = input("Triệu chứng: ").strip()
            
            if symptoms_input:
                selected_symptoms = [s.strip() for s in symptoms_input.split(',')]
                
                try:
                    result = predictor.predict_disease(selected_symptoms)
                    
                    print("\n" + "=" * 70)
                    print("KẾT QUẢ DỰ ĐOÁN")
                    print("=" * 70)
                    print(f"\n🎯 Bệnh được dự đoán: {result['primary_prediction']}")
                    print(f"📊 Độ tin cậy: {result['confidence']*100:.1f}%")
                    
                    print(f"\n✅ Triệu chứng khớp: {len(result['matched_symptoms'])}")
                    for symptom in result['matched_symptoms']:
                        print(f"   • {symptom}")
                    
                    if result['unmatched_symptoms']:
                        print(f"\n⚠️  Triệu chứng không tìm thấy: {len(result['unmatched_symptoms'])}")
                        for symptom in result['unmatched_symptoms']:
                            print(f"   • {symptom}")
                    
                    print(f"\n📈 Top 5 bệnh có khả năng:")
                    for i, pred in enumerate(result['top_predictions'], 1):
                        print(f"   {i}. {pred['disease']}: {pred['percentage']}")
                    
                except Exception as e:
                    print(f"\n❌ Lỗi khi dự đoán: {e}")
        
        print("\n" + "=" * 70)
        print("✅ HOÀN TẤT!")
        print("=" * 70)
        print("\n💡 Bạn có thể chạy web server bằng lệnh: python app.py")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()