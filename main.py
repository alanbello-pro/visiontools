"""
VisionTools - Menu Principal
Sistema interativo de ferramentas de visão computacional
"""

import os
from typing import List, Dict, Any

def limpar_tela():
    """Limpa a tela do terminal (cross-platform)."""
    os.system('cls' if os.name == 'nt' else 'clear')


def exibir_cabecalho():
    """Exibe o cabeçalho do menu principal."""
    print("\n" + "="*60)
    print(" "*15 + "🛠️  VISIONTOOLS  🛠️")
    print(" "*10 + "Ferramentas de Visão Computacional")
    print("="*60 + "\n")


def exibir_menu_ferramentas():
    """Exibe o menu de seleção de ferramentas."""
    print("FERRAMENTAS DISPONÍVEIS:\n")
    print(" [1] Analisador de Vídeo")
    print("  └─ Tracking, velocidade e análise de objetos\n")
    print(" [0] Sair")
    print("\n" + "-"*60)


def obter_opcao_usuario(mensagem: str, opcoes_validas: List[str]) -> str:
    """
    Solicita entrada do usuário com validação.
    
    Args:
        mensagem: Mensagem a ser exibida
        opcoes_validas: Lista de opções válidas
        
    Returns:
        str: Opção escolhida pelo usuário
    """
    while True:
        escolha = input(f"{mensagem}: ").strip()
        if escolha in opcoes_validas:
            return escolha
        print(f"Opção inválida! Escolha entre: {', '.join(opcoes_validas)}")


def obter_confirmacao(mensagem: str = "Confirmar") -> bool:
    """
    Solicita confirmação sim/não do usuário.
    
    Args:
        mensagem: Mensagem de confirmação
        
    Returns:
        bool: True se confirmado, False caso contrário
    """
    resposta = input(f"\n{mensagem}? (s/n): ").strip().lower()
    return resposta in ['s', 'sim', 'y', 'yes']


def configurar_analisador_video() -> Dict[str, Any]:
    """
    Menu de configuração interativo para o analisador de vídeo.
    
    Returns:
        dict: Dicionário com as configurações escolhidas
    """
    print("\n" + "="*60)
    print("CONFIGURAÇÃO - ANALISADOR DE VÍDEO")
    print("="*60 + "\n")
    
    config = {
        'config_path': 'config.json',
        'enable_csv': True,
        'enable_video': True,
        'show_window': False
    }
    
    # 1. Arquivo de configuração
    print("[1/4] Arquivo de Configuração")
    print("  Padrão: config.json")
    usar_padrao = obter_confirmacao("Usar arquivo padrão")
    
    if not usar_padrao:
        config['config_path'] = input("  Digite o caminho do config: ").strip()
    
    print(f"  ✓ Usando: {config['config_path']}\n")
    
    # 2. Modo de saída
    print("[2/4] Modo de Saída")
    print("[1] Ambos (CSV + Vídeo anotado)")
    print("[2] Apenas CSV")
    print("[3] Apenas Vídeo")
    print("[4] Nenhum (apenas tracking)")
    
    modo_saida = obter_opcao_usuario("  Escolha o modo", ['1', '2', '3', '4'])
    
    if modo_saida == '1':
        config['enable_csv'] = True
        config['enable_video'] = True
        print("  ✓ Gerando CSV + Vídeo\n")
    elif modo_saida == '2':
        config['enable_csv'] = True
        config['enable_video'] = False
        print("  ✓ Gerando apenas CSV\n")
    elif modo_saida == '3':
        config['enable_csv'] = False
        config['enable_video'] = True
        print("  ✓ Gerando apenas Vídeo\n")
    else:
        config['enable_csv'] = False
        config['enable_video'] = False
        print("  ✓ Modo tracking apenas\n")
    
    # 3. Exibição de janela
    print("[3/4] Exibição Durante Processamento")
    config['show_window'] = obter_confirmacao("Exibir janela de vídeo")
    
    status = "habilitada" if config['show_window'] else "desabilitada"
    print(f"  ✓ Janela {status}\n")
    
    # 4. Resumo
    print("="*60)
    print("RESUMO DAS CONFIGURAÇÕES:")
    print("="*60)
    print(f"  Config:     {config['config_path']}")
    print(f"  CSV:        {'✓ Sim' if config['enable_csv'] else '✗ Não'}")
    print(f"  Vídeo:      {'✓ Sim' if config['enable_video'] else '✗ Não'}")
    print(f"  Janela:     {'✓ Sim' if config['show_window'] else '✗ Não'}")
    print("="*60)
    
    return config


def executar_analisador_video(config: Dict[str, Any]):
    """
    Executa o analisador de vídeo com as configurações fornecidas.
    
    Args:
        config: Dicionário com configurações
    """
    print("\n Iniciando Analisador de Vídeo...\n")
    
    # Importa o módulo
    from src.tools.analisador_de_video import initialize_components, run_processing_loop
    
    # Cria argumentos simulando CLI
    class Args:
        def __init__(self, config_dict):
            self.config = config_dict['config_path']
            self.only_csv = config_dict['enable_csv'] and not config_dict['enable_video']
            self.only_video = config_dict['enable_video'] and not config_dict['enable_csv']
            self.no_csv = not config_dict['enable_csv']
            self.no_video = not config_dict['enable_video']
            self.show = config_dict['show_window']
            self.no_show = not config_dict['show_window']
    
    args = Args(config)
    
    try:
        # Inicializa componentes
        project_root = os.path.dirname(os.path.abspath(__file__))
        components = initialize_components(project_root=project_root, config_path=args.config, cli_args=args)
        
        # Executa processamento
        run_processing_loop(components)
        
        # Cleanup
        print("\n🧹 Liberando recursos...")
        if components['track_lifecycle_manager']:
            components['track_lifecycle_manager'].cleanup_all_tracking()
        components['resources'].cleanup()
        print("✅ Recursos liberados com sucesso")
        
    except KeyboardInterrupt:
        print("\n\n Processamento interrompido pelo usuário.")
    except Exception as e:
        print(f"\n Erro durante o processamento:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()


def menu_principal():
    """Loop principal do menu interativo."""
    while True:
        limpar_tela()
        exibir_cabecalho()
        exibir_menu_ferramentas()
        
        escolha = obter_opcao_usuario("\n👉 Digite o número da ferramenta", ['0', '1'])
        
        if escolha == '0':
            print("\n👋 Encerrando VisionTools. Até logo!")
            break
        
        elif escolha == '1':
            # Analisador de Vídeo
            limpar_tela()
            config = configurar_analisador_video()
            
            if obter_confirmacao("\n▶️  Executar agora"):
                executar_analisador_video(config)
                input("\n\n Pressione ENTER para voltar ao menu principal...")
            else:
                print(" Execução cancelada.")
                input("\nPressione ENTER para voltar ao menu...")


def main():
    """Ponto de entrada principal."""
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\n\n Programa encerrado pelo usuário. Até logo!")
    except Exception as e:
        print(f"\n Erro inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()