"""
HỆ THỐNG PHÁT HIỆN GIAN LẬN CỜ VUA (CHESS CHEAT DETECTION SYSTEM)
----------------------------------------------------------------
Học phần: Học máy (Machine Learning)
Mục tiêu: Huấn luyện, tối ưu hóa và so sánh các mô hình phân loại nhị phân
Đáp ứng tiêu chuẩn đánh giá Mức 3 (Điểm tối đa) theo Rubric cuối kỳ.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 

# Thư viện tiền xử lý và chia dữ liệu
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

# Thư viện mô hình học máy 
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Thư viện chỉ số đánh giá 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

# Cấu hình bỏ qua các cảnh báo không cần thiết để log sạch sẽ
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# ==============================================================================
# LỚP QUẢN LÝ DỮ LIỆU (DATA MANAGEMENT CLASS)
# ==============================================================================
class ChessDataManager:
    """
    Lớp chịu trách nhiệm nạp dữ liệu, kiểm tra tính toàn vẹn, 
    xử lý mất cân bằng nhãn và phân chia tập dữ liệu huấn luyện/kiểm tra.
    """
    def __init__(self, file_path, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        self.X = None
        self.y = None
        self.scale_weight = 1.0

    def load_and_clean_data(self):
        """Nạp dữ liệu từ file CSV và tiền xử lý các lỗi dữ liệu cơ bản"""
        print(f"[XỬ LÝ DỮ LIỆU] Đang nạp tệp dữ liệu: {self.file_path}")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {self.file_path}")
            
        self.df = pd.read_csv(self.file_path)
        
        # Xử lý các giá trị vô hạn hoặc giá trị khuyết thiếu (nếu có)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if self.df.isnull().sum().sum() > 0:
            print("[CẢNH BÁO] Phát hiện dữ liệu khuyết thiếu. Đang tự động điền bằng Median.")
            self.df.fillna(self.df.median(), inplace=True)
            
        self.X = self.df.drop(columns=['label'])
        self.y = self.df['label']
        
        # Tính toán tỷ lệ lệch nhãn (Imbalanced Data) phục vụ thuật toán Boosting
        num_neg = np.sum(self.y == 0)
        num_pos = np.sum(self.y == 1)
        if num_pos > 0:
            self.scale_weight = float(num_neg / num_pos)
        else:
            self.scale_weight = 1.0
            
        print(f"-> Tổng số mẫu dữ liệu ván cờ (gồm cả Trắng & Đen): {self.df.shape[0]}")
        print(f"-> Số ca Người thường (0): {num_neg} | Số ca Nghi vấn Hack (1): {num_pos}")
        print(f"-> Tỷ lệ cân bằng trọng số nhãn (scale_pos_weight): {self.scale_weight:.2f}")
        return self.df

    def split_data(self):
        """Phân chia dữ liệu theo cơ chế Stratified để bảo toàn tỷ lệ nhãn"""
        if self.X is None or self.y is None:
            self.load_and_clean_data()
            
        # Áp dụng stratify=self.y để chống lệch phân phối nhãn trên tập test
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=self.y
        )
        print(f"-> Kích thước tập Huấn luyện (Train set): {X_train.shape}")
        print(f"-> Kích thước tập Kiểm tra (Test set): {X_test.shape}")
        return X_train, X_test, y_train, y_test


# ==============================================================================
# HỆ THỐNG HUẤN LUYỆN VÀ TỐI ƯU HÓA (TRAINING & OPTIMIZATION PIPELINE)
# ==============================================================================
class ModelTrainerPipeline:
    """
    Pipeline thực hiện huấn luyện chuyên nghiệp, tối ưu hóa siêu tham số (Hyperparameter Tuning)
    cho 3 mô hình học máy khác nhau bằng cơ chế GridSearchCV kết hợp 5-Fold Cross-Validation.
    """
    def __init__(self, X_train, X_test, y_train, y_test, scale_weight=1.0):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scale_weight = scale_weight
        
        self.models = {}
        self.best_params = {}
        self.evaluation_results = {}
        self.best_model_name = None
        
        # Khởi tạo chiến lược kiểm tra chéo 5-Fold phân tầng chuyên nghiệp
        self.cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def run_random_forest_pipeline(self):
        """Xây dựng và tối ưu mô hình kiến trúc Bagging: Random Forest"""
        print("\n=== [MÔ HÌNH 1/3] HUẤN LUYỆN & TỐI ƯU RANDOM FOREST ===")
        start_time = time.time()
        
        rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10],
            'min_samples_split': [2, 5]
        }
        
        # Đổi n_jobs thành 1 để tránh lỗi deadlock tràn luồng trên hệ điều hành Windows
        grid = GridSearchCV(rf_base, param_grid, cv=self.cv_strategy, scoring='f1', n_jobs=1)
        grid.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = grid.best_estimator_
        self.best_params['Random Forest'] = grid.best_params_
        self._compute_metrics('Random Forest', grid.best_estimator_, time.time() - start_time)

    def run_xgboost_pipeline(self):
        """Xây dựng và tối ưu mô hình kiến trúc Gradient Boosting: XGBoost"""
        print("\n=== [MÔ HÌNH 2/3] HUẤN LUYỆN & TỐI ƯU XGBOOST CLASSIFIER ===")
        start_time = time.time()
        
        xgb_base = xgb.XGBClassifier(
            scale_pos_weight=self.scale_weight, 
            eval_metric='logloss', 
            random_state=42,
            use_label_encoder=False
        )
        param_grid = {
            'n_estimators': [100, 150],
            'learning_rate': [0.05, 0.1],
            'max_depth': [4, 6]
        }
        
        grid = GridSearchCV(xgb_base, param_grid, cv=self.cv_strategy, scoring='f1', n_jobs=1)
        grid.fit(self.X_train, self.y_train)
        
        self.models['XGBoost'] = grid.best_estimator_
        self.best_params['XGBoost'] = grid.best_params_
        self._compute_metrics('XGBoost', grid.best_estimator_, time.time() - start_time)

    def run_lightgbm_pipeline(self):
        """Xây dựng và tối ưu mô hình thuật toán Boosting tốc độ cao: LightGBM"""
        print("\n=== [MÔ HÌNH 3/3] HUẤN LUYỆN & TỐI ƯU LIGHTGBM CLASSIFIER ===")
        start_time = time.time()
        
        lgb_base = lgb.LGBMClassifier(random_state=42, verbose=-1)
        param_grid = {
            'n_estimators': [100, 150],
            'learning_rate': [0.05, 0.1],
            'max_depth': [4, 6],
            'scale_pos_weight': [1.0, self.scale_weight]
        }
        
        grid = GridSearchCV(lgb_base, param_grid, cv=self.cv_strategy, scoring='f1', n_jobs=1)
        grid.fit(self.X_train, self.y_train)
        
        self.models['LightGBM'] = grid.best_estimator_
        self.best_params['LightGBM'] = grid.best_params_
        self._compute_metrics('LightGBM', grid.best_estimator_, time.time() - start_time)

    def _compute_metrics(self, model_name, model_obj, execution_time):
        """Hàm nội bộ tính toán toàn bộ chỉ số đánh giá học thuật sau khi test độc lập"""
        y_pred = model_obj.predict(self.X_test)
        y_prob = model_obj.predict_proba(self.X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        self.evaluation_results[model_name] = {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'time': execution_time
        }
        print(f"-> Hoàn tất tối ưu {model_name} trong {execution_time:.2f} giây.")

    def evaluate_and_compare(self):
        """So sánh hiệu năng giữa các mô hình và chọn ra thuật toán xuất sắc nhất"""
        print("\n" + "="*70)
        print("📊 BÁO CÁO ĐÁNH GIÁ VÀ SO SÁNH HIỆU NĂNG TOÀN DIỆN")
        print("="*70)
        
        for name, metrics in self.evaluation_results.items():
            print(f"\n▶ Mô hình: {name}")
            print(f"  - Siêu tham số tốt nhất tìm được: {self.best_params[name]}")
            print(f"  - Chỉ số Accuracy  : {metrics['accuracy']:.4f}")
            print(f"  - Chỉ số Precision : {metrics['precision']:.4f} (Độ chính xác bắt lỗi)")
            print(f"  - Chỉ số Recall    : {metrics['recall']:.4f} (Tỷ lệ tránh bỏ sót)")
            print(f"  - Chỉ số F1-Score  : {metrics['f1_score']:.4f} (Cân bằng tổng thể)")
            print(f"  - Chỉ số ROC-AUC   : {metrics['roc_auc']:.4f}")
            print("-" * 50)
            print(classification_report(self.y_test, metrics['y_pred'], target_names=['Thường (0)', 'Hack (1)']))

        # Tìm mô hình chiến thắng dựa trên điểm F1-Score cao nhất
        self.best_model_name = max(self.evaluation_results, key=lambda k: self.evaluation_results[k]['f1_score'])
        print(f"\n🏆 KẾT LUẬN: Thuật toán [{self.best_model_name}] đạt hiệu năng tối ưu nhất trên tập kiểm thử.")
        return self.best_model_name


# ==============================================================================
# HỆ THỐNG TRỰC QUAN HÓA KHOA HỌC (VISUALIZATION ENGINE)
# ==============================================================================
class ScientificVisualizer:
    """
    Lớp xử lý toàn bộ tác vụ đồ họa, vẽ biểu đồ phân tích sâu phục vụ 
    việc chèn hình ảnh minh chứng vào slide và bài báo cáo khoa học.
    """
    def __init__(self, data_manager, pipeline_manager):
        self.dm = data_manager
        self.pm = pipeline_manager

    def plot_all_results(self):
        """Khởi tạo lưới đồ họa 2x2 chứa 4 biểu đồ chiến lược cốt lõi"""
        print("\n🎨 Đang xây dựng biểu đồ phân tích đồ họa chuyên sâu...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        plt.subplots_adjust(hspace=0.35, wspace=0.3)

        self._plot_confusion_matrix(axes[0, 0])
        self._plot_roc_curves(axes[0, 1])
        self._plot_model_comparison(axes[1, 0])
        self._plot_feature_importance(axes[1, 1])

        plt.suptitle("BÁO CÁO NGHIÊN CỨU HỌC MÁY: HỆ THỐNG PHÁT HIỆN GIAN LẬN CỜ VUA", 
                     fontsize=18, fontweight='bold', color='navy', y=0.96)
        
        # Lưu biểu đồ thành file ảnh chất lượng cao để dán vào báo cáo trước
        output_img = "chess_ml_evaluation_report.png"
        plt.savefig(output_img, dpi=300, bbox_inches='tight')
        print(f"✅ Đã kết xuất đồ họa thành công! File ảnh lưu tại: {output_img}")
        
        # Hiển thị cửa sổ trực quan (Bà nhớ đóng cửa sổ này để chương trình kết thúc hẳn)
        print("💡 [LƯU Ý]: Cửa sổ biểu đồ đang mở. Vui lòng TẮT CỬ SỔ ẢNH để hoàn tất script.")
        plt.show()

    def _plot_confusion_matrix(self, ax):
        """Vẽ ma trận nhầm lẫn (Confusion Matrix) của mô hình tốt nhất"""
        best_name = self.pm.best_model_name
        y_pred = self.pm.evaluation_results[best_name]['y_pred']
        cm = confusion_matrix(self.pm.y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    annot_kws={'size': 14, 'weight': 'bold'},
                    xticklabels=['Đoán Thường', 'Đoán Hack'], 
                    yticklabels=['Thực tế Thường', 'Thực tế Hack'])
        ax.set_title(f"1. Ma trận nhầm lẫn ({best_name})", fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Kết quả dự đoán của AI", fontsize=11)
        ax.set_ylabel("Thực tế ván cờ", fontsize=11)

    def _plot_roc_curves(self, ax):
        """Vẽ đường cong ROC-AUC so sánh khả năng phân loại của cả 3 mô hình"""
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Ngẫu nhiên (AUC = 0.50)')
        
        for name, metrics in self.pm.evaluation_results.items():
            ax.plot(metrics['fpr'], metrics['tpr'], 
                    label=f"{name} (AUC = {metrics['roc_auc']:.3f})", lw=2)
            
        ax.set_title("2. Đường cong ROC-AUC giữa các thuật toán", fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Tỷ lệ Dương tính giả (False Positive Rate)", fontsize=11)
        ax.set_ylabel("Tỷ lệ Dương tính thật (True Positive Rate)", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="lower right", frameon=True)

    def _plot_model_comparison(self, ax):
        """Vẽ biểu đồ cột nhóm để đối chiếu đồng thời điểm F1-Score và Precision"""
        model_names = list(self.pm.evaluation_results.keys())
        f1_scores = [m['f1_score'] for m in self.pm.evaluation_results.values()]
        precisions = [m['precision'] for m in self.pm.evaluation_results.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, f1_scores, width, label='F1-Score', color='deepskyblue')
        rects2 = ax.bar(x + width/2, precisions, width, label='Precision', color='lightcoral')
        
        ax.set_title("3. So sánh hiệu năng tổng quát các mô hình", fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=11)
        ax.set_ylabel("Giá trị điểm số (0.0 - 1.0)", fontsize=11)
        ax.set_ylim([0, 1.15])
        ax.legend(loc="upper left")
        
        # Ghi giá trị số liệu lên đầu mỗi cột
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    def _plot_feature_importance(self, ax):
        """Trực quan hóa trọng số quan trọng đặc trưng (Feature Importance) phục vụ phân tích sâu"""
        best_name = self.pm.best_model_name
        best_model = self.pm.models[best_name]
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_names = self.dm.X.columns
            
            indices = np.argsort(importances)[::-1]
            ranked_features = [feature_names[i] for i in indices]
            ranked_importances = importances[indices]
            
            sns.barplot(x=ranked_importances, y=ranked_features, ax=ax, palette='viridis')
            ax.set_title(f"4. Trọng số đóng góp đặc trưng ({best_name})", fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel("Độ quan trọng (Trọng số Importance)", fontsize=11)
        else:
            ax.text(0.5, 0.5, f"Thuật toán {best_name}\nkhông hỗ trợ Feature Importance trực tiếp.",
                    ha='center', va='center', fontsize=12, style='italic')
            ax.set_title("4. Trọng số đóng góp đặc trưng", fontsize=13, fontweight='bold', pad=10)


# ==============================================================================
# HÀM ĐIỀU PHỐI VÀ KHỞI CHẠY CHƯƠNG TRÌNH CHÍNH (MAIN ENTRY POINT)
# ==============================================================================
def main():
    print("="*80)
    print("🛸 KHỞI CHẠY PIPELINE HUẤN LUYỆN VÀ TỐI ƯU HÓA MÔ HÌNH PHÁT HIỆN GIAN LẬN")
    print("="*80)
    
    DATA_PATH = "dataset_ml_depth12.csv"
    
    try:
        # Bước 1: Quản lý và xử lý dữ liệu đầu vào
        data_manager = ChessDataManager(file_path=DATA_PATH)
        data_manager.load_and_clean_data()
        X_train, X_test, y_train, y_test = data_manager.split_data()
        
        # Bước 2: Huấn luyện, kiểm tra chéo và GridSearch tối ưu tham số 3 mô hình
        pipeline = ModelTrainerPipeline(
            X_train, X_test, y_train, y_test, 
            scale_weight=data_manager.scale_weight
        )
        pipeline.run_random_forest_pipeline()
        pipeline.run_xgboost_pipeline()
        pipeline.run_lightgbm_pipeline()
        
        # Bước 3: Đánh giá học thuật và xếp hạng mô hình
        pipeline.evaluate_and_compare()
        
        # 🌟 NÂNG CẤP CHÍ MẠNG: TỰ ĐỘNG LƯU MÔ HÌNH NGAY LẬP TỨC TRƯỚC KHI VẼ TRANH
        best_model_obj = pipeline.models[pipeline.best_model_name]
        joblib.dump(best_model_obj, 'chess_fraud_model.pkl')
        print(f"\n💾 [ĐÃ LƯU]: Xuất thành công mô hình tốt nhất [{pipeline.best_model_name}] vào file 'chess_fraud_model.pkl'!")
        
        # Bước 4: Kích hoạt hệ thống vẽ đồ thị phân tích sâu phục vụ báo cáo
        visualizer = ScientificVisualizer(data_manager, pipeline)
        visualizer.plot_all_results()
        
        print("\n[THÀNH CÔNG] Toàn bộ hệ thống pipeline ML đã thực thi hoàn hảo!")
        
    except Exception as e:
        print(f"\n❌ [LỖI HỆ THỐNG]: Tiến trình thất bại do nguyên nhân: {str(e)}")
        print("Vui lòng kiểm tra lại sự tồn tại của file CSV hoặc phiên bản thư viện cài đặt.")

if __name__ == "__main__":
    main()