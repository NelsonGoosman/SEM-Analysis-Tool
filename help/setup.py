import os
import sys
import subprocess
import venv
import platform

def main():
    # Determine the root of the project (assuming this script is in help/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Configuration
    venv_name = "sem_venv"  # Using the existing folder name
    venv_dir = os.path.join(project_root, venv_name)
    requirements_file = os.path.join(project_root, "requirements.txt")

    print(f"--- Setting up {venv_name} ---")

    # 1. Create Virtual Environment
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment at: {venv_dir}")
        venv.create(venv_dir, with_pip=True)
    else:
        print(f"Virtual environment already exists at: {venv_dir}")

    # 2. Locate executables
    if platform.system() == "Windows":
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
        activation_instruction = f".\\{venv_name}\\Scripts\\Activate.ps1"
    else:
        python_exe = os.path.join(venv_dir, "bin", "python")
        activation_instruction = f"source {venv_name}/bin/activate"

    if not os.path.exists(python_exe):
        print(f"Error: Python executable not found at {python_exe}")
        return

    # 3. Install Requirements
    if os.path.exists(requirements_file):
        print(f"Installing dependencies from: {requirements_file}")
        try:
            # Upgrade pip
            subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
            # Install requirements
            subprocess.check_call([python_exe, "-m", "pip", "install", "-r", requirements_file])
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            return
    else:
        print(f"Warning: requirements.txt not found at {requirements_file}")

    # 4. Activation Instructions
    print("\n" + "="*40)
    print("Setup Complete!")
    print("To activate the virtual environment, run:")
    print(f"  {activation_instruction}")
    print("="*40)

    print("To run the application, enter the following command: python ./src/app/main.py")

if __name__ == "__main__":
    main()
