import numpy as np
from typing import Optional

Ponto = tuple[float, float]
Trapezio = tuple[Ponto, Ponto, Ponto, Ponto]
Malha = list[list[Ponto]]


class PerspectiveCalibrator:

    def __init__(self, trapezio: Trapezio):

        if len(trapezio) != 4:
            raise ValueError("O trapézio de entrada deve conter exatamente 4 pontos.")
        self.trapezio = trapezio
        self.topo_esq, self.topo_dir, self.base_dir, self.base_esq = trapezio

        print("Calculando pontos de fuga...")
        self.pf_lateral, self.pf_horizontal = self._encontrar_pontos_de_fuga()

        if self.pf_lateral is None:
            raise ValueError("Lados laterais são paralelos. Não é possível calcular o ponto de fuga lateral.")
        if self.pf_horizontal is None:
            raise ValueError("Lados horizontais são paralelos. Não é possível calcular o ponto de fuga horizontal.")

        print(f"PF Lateral: {self.pf_lateral}")
        print(f"PF Horizontal: {self.pf_horizontal}")

    @staticmethod
    def _calcular_interseccao_linhas(p1: Ponto, p2: Ponto, p3: Ponto, p4: Ponto) -> Optional[Ponto]:
        linha1 = np.cross([*p1, 1], [*p2, 1])
        linha2 = np.cross([*p3, 1], [*p4, 1])
        x, y, w = np.cross(linha1, linha2)

        if np.isclose(w, 0):
            return None 

        return float(x / w), float(y / w)

    def _encontrar_pontos_de_fuga(self) -> tuple[Optional[Ponto], Optional[Ponto]]:
        pf_lateral = self._calcular_interseccao_linhas(self.base_esq, self.topo_esq, self.base_dir, self.topo_dir)
        pf_horizontal = self._calcular_interseccao_linhas(self.topo_esq, self.topo_dir, self.base_esq, self.base_dir)
        return pf_lateral, pf_horizontal

    def _interpolar_pontos_projetivos(
        self, p_inicial: Ponto, p_final: Ponto, p_fuga: Ponto, 
        num_pontos_base: int, indice_inicio: int, indice_fim: int
    ) -> list[Ponto]:
        p_ini = np.array(p_inicial, dtype=float)
        p_fim = np.array(p_final, dtype=float)
        p_fuga_np = np.array(p_fuga, dtype=float)

        pontos_calculados = []
        if num_pontos_base < 2:
            if num_pontos_base == 1 and indice_inicio == 0 and indice_fim == 0:
                return [tuple(p_ini.astype(int))]
            return []

        num_intervalos_base = num_pontos_base - 1
        v_start = p_ini - p_fuga_np
        v_end = p_fim - p_fuga_np
        norm_v_start = np.linalg.norm(v_start)
        norm_v_end = np.linalg.norm(v_end)

        if np.isclose(norm_v_start, 0) and np.isclose(norm_v_end, 0):
            return [p_inicial] * (indice_fim - indice_inicio + 1)

        for i in range(indice_inicio, indice_fim + 1):
            t = i / num_intervalos_base
            weight_start = norm_v_end * (1 - t)
            weight_end = norm_v_start * t
            denominador = weight_start + weight_end

            if np.isclose(denominador, 0):
                ponto_i = p_ini if t < 0.5 else p_fim
            else:
                ponto_i = (p_ini * weight_start + p_fim * weight_end) / denominador
            pontos_calculados.append(tuple(ponto_i.astype(int)))
        return pontos_calculados

    def calcular_malha(self, num_linhas: int, num_colunas: int, linhas_antes: int = 0, linhas_depois: int = 0) -> Malha:
        print(f"Calculando Malha Projetiva...")
        indice_linha_inicio = -linhas_antes
        indice_linha_fim = (num_linhas - 1) + linhas_depois

        print(f"Calculando colunas laterais expandidas (Índices de {indice_linha_inicio} a {indice_linha_fim})...")

        pontos_esquerda = self._interpolar_pontos_projetivos(self.base_esq, self.topo_esq, self.pf_lateral, num_linhas, indice_linha_inicio, indice_linha_fim)
        pontos_direita = self._interpolar_pontos_projetivos(self.base_dir, self.topo_dir, self.pf_lateral, num_linhas, indice_linha_inicio, indice_linha_fim)

        total_linhas_finais = len(pontos_esquerda)
        print(f"  Construindo malha final de {total_linhas_finais}x{num_colunas} colunas...")
        malha_final = []
        for p_esq, p_dir in zip(pontos_esquerda, pontos_direita):
            linha_horizontal = self._interpolar_pontos_projetivos(p_esq, p_dir, self.pf_horizontal, num_colunas, 0, num_colunas - 1)
            if linha_horizontal:
                malha_final.append(linha_horizontal)

        print(f"Cálculo da Malha ({len(malha_final)}x{num_colunas}) concluído.")
        return malha_final

