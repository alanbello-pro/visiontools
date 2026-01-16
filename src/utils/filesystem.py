import os
from datetime import datetime

def verificar_e_criar_diretorios_input(project_root: str, base_dir_name="input", subdirs=["videos", "models"]):

    base_dir = os.path.join(project_root, base_dir_name)
    diretorios_criados = False
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        diretorios_criados = True

    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
            diretorios_criados = True

    if diretorios_criados:
        raise FileNotFoundError(
            f"O diretório de entrada '{base_dir}' e/ou seus subdiretórios ('videos', 'models') não existiam e foram criados. "
            f"Por favor, adicione os vídeos de origem em '{os.path.join(base_dir, 'videos')}' "
            f"e os modelos YOLO (.pt) em '{os.path.join(base_dir, 'models')}' e execute o programa novamente."
        )

def criar_diretorio_de_saida(project_root: str, base_dir_name="output"):
    
    base_dir = os.path.join(project_root, base_dir_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nDiretório de saída criado em: {output_dir}")
    return output_dir
