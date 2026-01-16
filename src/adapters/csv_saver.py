import csv
import pandas as pd
from typing import List, Tuple

Ponto = Tuple[float, float]
Malha = List[List[Ponto]]

class CsvSaver:
    FIELDNAMES = [
        'frame_id', 'track_id', 'classe', 'confianca',
        'box_x1', 'box_y1', 'box_x2', 'box_y2', 'velocidade_kmh', 'faixa', 'cor_dominante_bgr'
    ]

    def __init__(self, arquivo_saida, batch_size):
        self.arquivo_saida = arquivo_saida
        self.csvfile = None
        self.writer = None
        self.buffer = []
        self.batch_size = batch_size
        self.saved_track_ids = set()
        self._iniciar_arquivo()

    def _iniciar_arquivo(self):
        try:
            self.csvfile = open(self.arquivo_saida, 'w', newline='', encoding='utf-8')
            self.writer = csv.DictWriter(self.csvfile, fieldnames=self.FIELDNAMES)
            self.writer.writeheader()
            print(f"Arquivo CSV '{self.arquivo_saida}' aberto para escrita.")
        except IOError as e:
            print(f"Erro ao abrir o arquivo CSV: {e}")

    def _converter_valor(self, valor):
        if hasattr(valor, 'item'):
            return float(valor.item())
        try:
            return float(valor)
        except (ValueError, TypeError):
            return valor

    def _flush(self):
        if not self.writer or not self.buffer:
            return
        try:
            self.writer.writerows(self.buffer)
            self.buffer = []
        except IOError as e:
            print(f"Erro ao escrever lote no CSV: {e}")

    def salvar_registro(self, registro):
        if not self.writer:
            print("Erro: O arquivo CSV não está aberto para escrita.")
            return
        
        cor_dominante = registro.get('cor_dominante_bgr')
        track_id = int(self._converter_valor(registro.get('track_id', '')))
        self.saved_track_ids.add(track_id)

        cor_formatada = ''
        if isinstance(cor_dominante, (list, tuple)) and len(cor_dominante) == 3:
            cor_formatada = f"{cor_dominante[0]},{cor_dominante[1]},{cor_dominante[2]}"

        registro_processado = {
            'frame_id': int(self._converter_valor(registro.get('frame_id', ''))),
            'track_id': int(self._converter_valor(registro.get('track_id', ''))),
            'classe': registro.get('classe', ''),
            'confianca': round(self._converter_valor(registro.get('confianca', 0.0)), 4),
            'box_x1': round(self._converter_valor(registro.get('box_x1', 0.0)), 2),
            'box_y1': round(self._converter_valor(registro.get('box_y1', 0.0)), 2),
            'box_x2': round(self._converter_valor(registro.get('box_x2', 0.0)), 2),
            'box_y2': round(self._converter_valor(registro.get('box_y2', 0.0)), 2),
            'velocidade_kmh': round(self._converter_valor(registro.get('velocidade_kmh', 0.0)), 2) if registro.get('velocidade_kmh') is not None else '',
            'faixa': registro.get('faixa', ''),
            'cor_dominante_bgr': cor_formatada
        }

        self.buffer.append(registro_processado)
        if len(self.buffer) >= self.batch_size:
            self._flush()

    def get_unique_track_count(self) -> int:
        """Retorna o número de IDs de track únicos que foram salvos."""
        return len(self.saved_track_ids)

    def is_ready(self) -> bool:
        """Verifica se o writer está pronto para escrever no arquivo."""
        return self.writer is not None and self.csvfile is not None

    def close(self):
        """Escreve os registros restantes e fecha o arquivo CSV."""
        self._flush() # Garante que o último lote seja escrito
        if self.csvfile:
            self.csvfile.close()
            print(f"Arquivo CSV '{self.arquivo_saida}' fechado.")

def exportar_malha_para_csv(malha: Malha, espacamento_linhas_mm: float, espacamento_colunas_mm: float, nome_arquivo_csv: str, csv_round_precision: int, csv_separator: str):
    
    print(f"Exportando malha para '{nome_arquivo_csv}'...")
    dados_para_csv = []
    
    if not malha:
        print("Erro: Malha está vazia, nada para exportar.")
        return

    for i, linha in enumerate(malha):
        milimetros_y = i * espacamento_linhas_mm
        
        for j, ponto_pixel in enumerate(linha):
            milimetros_x = j * espacamento_colunas_mm
            
            x_pixel, y_pixel = ponto_pixel
            
            dados_para_csv.append({
                'linha_id': i,
                'coluna_id': j,
                'x_pixel': x_pixel,
                'y_pixel': y_pixel,
                'milimetros_x': round(milimetros_x, csv_round_precision),
                'milimetros_y': round(milimetros_y, csv_round_precision)
            })

    try:
        df = pd.DataFrame(dados_para_csv)
        df.to_csv(nome_arquivo_csv, index=False, sep=csv_separator)
        print(f"Sucesso! Malha com {len(df)} pontos salva em '{nome_arquivo_csv}'.")
    except Exception as e:
        print(f"Erro ao salvar o arquivo CSV: {e}")