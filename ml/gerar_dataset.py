import argparse
import itertools
import random
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import get_settings
from app.schemas import Categoria, Prioridade
from app.services.preprocess import normalize_text, strip_accents

SEED = 42
SAMPLES_PER_CELL = 40
TEST_SIZE = 0.2

M, E, A = Categoria.MECANICA, Categoria.ELETRICA, Categoria.AUTOMACAO
BAIXA, MEDIA, ALTA = Prioridade.BAIXA, Prioridade.MEDIA, Prioridade.ALTA

COMPONENTES = {
    M: [
        "rolamento do motor da esteira", "redutor da esteira transportadora", "acoplamento da bomba hidráulica",
        "correia do ventilador", "eixo do misturador", "bomba de óleo", "válvula pneumática",
        "cilindro hidráulico", "compressor de ar", "mancal do exaustor", "corrente do elevador de canecas",
        "esteira da linha 1",
    ],
    E: [
        "painel elétrico da linha 2", "disjuntor do motor principal", "contator da bomba",
        "transformador da subestação", "quadro de distribuição", "VFD da esteira", "cabo de alimentação do motor",
        "motor trifásico da bomba", "fonte 24V do painel", "relé térmico do exaustor",
        "iluminação da área de envase", "tomada industrial do setor de embalagem",
    ],
    A: [
        "CLP da linha de envase", "IHM da prensa", "sensor indutivo da esteira", "VFD da bomba (parametrização)",
        "encoder do eixo principal", "transmissor de pressão 4-20mA", "rede Profibus do painel",
        "servo drive da embaladora", "sensor fotoelétrico da seladora", "módulo de I/O remoto",
        "sistema SCADA da planta", "válvula proporcional",
    ],
}

SINTOMAS = {
    (M, ALTA): [
        "com vazamento de óleo no piso e risco de queda", "travou e parou a linha inteira",
        "soltando fumaça por atrito excessivo", "com vazamento de gás refrigerante",
        "quebrou e há risco de acidente", "com peça solta e risco para o operador",
    ],
    (M, MEDIA): [
        "com ruído anormal e vibração", "aquecendo acima do normal", "com folga excessiva no eixo",
        "perdendo pressão aos poucos", "com desgaste visível", "trabalhando com vazamento leve de óleo",
    ],
    (M, BAIXA): [
        "para lubrificação programada", "para inspeção preventiva mensal", "com troca preventiva agendada",
        "para reaperto de parafusos na próxima parada", "para limpeza e inspeção de rotina",
        "para ajuste de alinhamento preventivo",
    ],
    (E, ALTA): [
        "com cheiro de queimado e faíscas", "apresentando risco de choque elétrico", "soltando fumaça",
        "em curto-circuito derrubando tudo", "com cabo exposto e risco de choque", "pegou fogo",
    ],
    (E, MEDIA): [
        "desarmando com frequência", "com aquecimento acima do normal nos terminais", "com queda de tensão",
        "oscilando a corrente", "com ruído de zumbido", "com fusível queimando repetidamente",
    ],
    (E, BAIXA): [
        "para termografia preventiva", "para reaperto de bornes programado", "para inspeção elétrica de rotina",
        "com troca preventiva de lâmpadas", "para medição de isolamento periódica",
        "para identificação dos circuitos",
    ],
    (A, ALTA): [
        "perdeu comunicação e parou a linha inteira", "acionou parada de emergência sem causa aparente",
        "com falha no intertravamento de segurança", "com comando travado e máquina operando sem controle",
        "com falha crítica e risco de acidente", "com falso sinal de emergência",
    ],
    (A, MEDIA): [
        "com erro E04 intermitente", "com leitura 4-20mA oscilando", "perdendo comunicação de forma intermitente",
        "com alarme de falha de encoder", "travando e reiniciando sozinho", "com sensor falhando de vez em quando",
    ],
    (A, BAIXA): [
        "para backup do programa agendado", "para atualização de firmware na próxima parada",
        "para calibração periódica", "para ajuste de parâmetros de rotina",
        "para revisão da lógica de baixa criticidade", "para inclusão de novo alarme na IHM",
    ],
}

CONTEXTOS = [
    "no turno da noite", "desde ontem", "na linha 3", "no setor de envase", "durante a partida",
    "após a última manutenção", "na área de expedição", "no turno da manhã", "no setor de utilidades",
    "após a troca de turno", "",
]

FRAMES_CORRETIVOS = [
    "{c} {s} {ctx}", "{ctx}: {c} {s}", "Operador informa que {c} {s} {ctx}", "Favor verificar: {c} {s}. {ctx}",
]
FRAMES_PREVENTIVOS = ["{c} {s} {ctx}", "Preventiva: {c} {s} {ctx}", "Agendar {c} {s}. {ctx}"]

AMBIGUOS = [
    ("motor desarmando", (E, M), MEDIA),
    ("motor esquentando e desarmando o térmico", (E, M), MEDIA),
    ("esteira parando sem aviso", (A, M), MEDIA),
    ("bomba não parte de manhã", (E, A), MEDIA),
    ("inversor dando falha na partida do motor", (A, E), MEDIA),
    ("equipamento desligando sozinho", (A, E), MEDIA),
    ("ventilador parando de vez em quando", (M, E), MEDIA),
    ("motor desarmando com cheiro de queimado", (E, M), ALTA),
    ("painel esquentando e motor travando", (E, M), MEDIA),
    ("máquina parando e reiniciando", (A, M), MEDIA),
    ("vibração e queda de rotação no motor", (M, E), MEDIA),
    ("sensor e motor sem resposta", (A, E), MEDIA),
]
AMBIGUOS_CONTEXTOS_POR_ITEM = 4
AMBIGUOS_PROBABILIDADE_PRIMARIA = 0.6


def tidy(text: str) -> str:
    text = re.sub(r"^\s*[:,]\s*", "", text)
    text = re.sub(r"\s+([.:,])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[0].upper() + text[1:] if text else text


def add_typo(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    candidates = [index for index, word in enumerate(words) if len(word) > 5 and word.isalpha()]
    if not candidates:
        return text
    index = rng.choice(candidates)
    word = words[index]
    position = rng.randrange(1, len(word) - 2)
    if rng.random() < 0.5:
        word = word[:position] + word[position + 1] + word[position] + word[position + 2 :]
    else:
        word = word[:position] + word[position + 1 :]
    words[index] = word
    return " ".join(words)


def vary_surface(text: str, rng: random.Random) -> str:
    if rng.random() < 0.15:
        text = add_typo(text, rng)
    if rng.random() < 0.30:
        text = strip_accents(text)
    if rng.random() < 0.08:
        text = text.upper()
    return text


def generate_cell(
    categoria: Categoria, prioridade: Prioridade, quantity: int, rng: random.Random, seen: set[str]
) -> list[dict[str, str]]:
    frames = FRAMES_PREVENTIVOS if prioridade == BAIXA else FRAMES_CORRETIVOS
    combinations = list(
        itertools.product(COMPONENTES[categoria], SINTOMAS[(categoria, prioridade)], CONTEXTOS, frames)
    )
    rng.shuffle(combinations)
    rows = []
    for componente, sintoma, contexto, frame in combinations:
        text = vary_surface(tidy(frame.format(c=componente, s=sintoma, ctx=contexto)), rng)
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"texto": text, "categoria": categoria.value, "prioridade": prioridade.value})
        if len(rows) == quantity:
            break
    return rows


def generate_ambiguous(rng: random.Random, seen: set[str]) -> list[dict[str, str]]:
    rows = []
    for base, (primary, secondary), prioridade in AMBIGUOS:
        for contexto in rng.sample(CONTEXTOS[:-1], AMBIGUOS_CONTEXTOS_POR_ITEM):
            categoria = primary if rng.random() < AMBIGUOS_PROBABILIDADE_PRIMARIA else secondary
            text = vary_surface(tidy(f"{base} {contexto}"), rng)
            key = normalize_text(text)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"texto": text, "categoria": categoria.value, "prioridade": prioridade.value})
    return rows


def build_dataset(seed: int = SEED, samples_per_cell: int = SAMPLES_PER_CELL) -> pd.DataFrame:
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for categoria in Categoria:
        for prioridade in Prioridade:
            rows.extend(generate_cell(categoria, prioridade, samples_per_cell, rng, seen))
    rows.extend(generate_ambiguous(rng, seen))
    frame = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return frame


def split_dataset(frame: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = frame["categoria"] + "|" + frame["prioridade"]
    train, test = train_test_split(frame, test_size=TEST_SIZE, stratify=strata, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def gerar_e_salvar(data_dir: Path, seed: int = SEED, forcar: bool = False) -> dict[str, int]:
    frozen_path = data_dir / "test_frozen.csv"
    if frozen_path.exists() and not forcar:
        raise FileExistsError(
            f"{frozen_path} já existe e está congelado. Use --forcar para regenerar (invalida benchmarks anteriores)."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = build_dataset(seed)
    train, test = split_dataset(frame, seed)
    frame.to_csv(data_dir / "chamados.csv", index=False, encoding="utf-8")
    train.to_csv(data_dir / "train.csv", index=False, encoding="utf-8")
    test.to_csv(frozen_path, index=False, encoding="utf-8")
    return {"total": len(frame), "treino": len(train), "teste_congelado": len(test)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--forcar", action="store_true")
    parser.add_argument("--saida", type=Path, default=get_settings().data_dir)
    args = parser.parse_args()
    print(gerar_e_salvar(args.saida, args.seed, args.forcar))


if __name__ == "__main__":
    main()
