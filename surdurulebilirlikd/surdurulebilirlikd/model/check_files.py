import os

def check_model_files():
    files_to_check = [
        'model.pkl',
        'model_dropout.h5',
        'scaler.pkl'
    ]
    
    model_dir = 'model'
    for file in files_to_check:
        path = os.path.join(model_dir, file)
        if os.path.exists(path):
            print(f"✓ {file} mevcut")
        else:
            print(f"✗ {file} bulunamadı!")

if __name__ == "__main__":
    check_model_files() 