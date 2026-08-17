import os
import sys
import threading
import time
import webbrowser


def open_browser(url: str) -> None:
    time.sleep(1.5)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sleep_app.settings')
    url = 'http://127.0.0.1:8000/'
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    # Ensure model exists; if not, train automatically
    try:
        from pathlib import Path
        import joblib
        base = Path(__file__).resolve().parent
        model_path = base / 'models' / 'sleep_model.joblib'
        if not model_path.exists():
            print('Model not found. Training now...')
            import train_model  # triggers training when called below
            train_model.train_and_save_model()
            print('Training completed.')
    except Exception as e:
        print('Warning: could not auto-train model:', e)

    from django.core.management import execute_from_command_line
    sys.argv = ['manage.py', 'runserver', '127.0.0.1:8000']
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()


