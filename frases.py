
import json
import os
import time
import unicodedata
import urllib.error
import urllib.request

import spacy


# ============================================================
# CONFIGURAÇÕES DOS MODELOS
# ============================================================

# Groq: modelo principal
GROQ_MODELO = os.environ.get(
    "GROQ_MODELO",
    "openai/gpt-oss-20b"
).strip()

# Gemini: modelo reserva
GEMINI_MODELO = os.environ.get(
    "GEMINI_MODELO",
    "gemini-3.8-flash"
).strip()

# Modelos Gemini adicionais, separados por vírgula.
# Serão tentados caso o Gemini principal também falhe.
GEMINI_RESERVA = [
    modelo.strip()
    for modelo in os.environ.get("GEMINI_MODELO_RESERVA", "").split(",")
    if modelo.strip()
]


# ============================================================
# CARREGAMENTO DO SPACY
# ============================================================

nlp = spacy.load("pt_core_news_sm")


# ============================================================
# VOCABULÁRIO E CLASSES DOS SINAIS
# ============================================================

# Vocabulário fechado do modelo dinâmico:
# classe de cada sinal (chave sem acento e em minúsculas).

CLASSES = {
    "aluno": "pessoa",
    "filho": "pessoa",

    "america": "lugar",
    "banco": "lugar",
    "banheiro": "lugar",
    "esquina": "lugar",

    "maca": "fruta",
    "bala": "doce",
    "espelho": "objeto",
    "vacina": "objeto",

    "sapo": "animal",
    "barulho": "substantivo",

    "amarelo": "adjetivo (cor)",
    "ruim": "adjetivo",
    "cinco": "numeral",

    "medo": "sentimento (quem sente e uma pessoa)",
    "vontade": "sentimento (quem sente e uma pessoa)",

    "acontecer": "verbo",
    "aproveitar": "verbo",
    "conhecer": "verbo",
}

SENTIMENTOS = {
    g for g, c in CLASSES.items()
    if c.startswith("sentimento")
}


# ============================================================
# PROMPT DE TRADUÇÃO
# ============================================================

PROMPT = """Voce converte sinais de Libras, na ordem em que foram feitos, em uma frase curta e natural em portugues.

Regras:
- Use TODAS as palavras, sem mudar o sentido e sem inventar informacao nova.
- Pode acrescentar verbos (ter, ser, estar, ir...), artigos e preposicoes, e flexionar os verbos.
- Em Libras a ordem nao segue o portugues: o primeiro sinal costuma ser o assunto ou o objeto, nao necessariamente o sujeito.
- Sentimentos (medo, vontade) sao sentidos por pessoas, nunca por lugares, objetos ou animais. Escolha sempre o sujeito e o objeto mais plausiveis no dia a dia.
- Palavras em MAIUSCULAS foram soletradas no alfabeto manual (nomes ou siglas): mantenha-as exatamente como estao, apenas com inicial maiuscula.
- Responda so com a frase, terminada em ponto final (ou em ? se for pergunta).

Exemplos:
Sinais: aluno medo conhecer américa
Frase: O aluno tem medo de conhecer a América.

Sinais: américa medo filho
Frase: O filho tem medo da América.

Sinais: vacina medo filho
Frase: O filho tem medo da vacina.

Sinais: filho vontade banheiro
Frase: O filho tem vontade de ir ao banheiro.

Sinais: maçã amarelo
Frase: A maçã é amarela.

Sinais: banco esquina
Frase: O banco fica na esquina.

Sinais: MARIA filho medo
Frase: A Maria tem medo do filho.

Sinais: {glosas}
Analise: {analise}
Frase:"""


# Cache: mesma sequência de sinais -> mesma frase
CACHE = {}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def sem_acento(texto):
    """Remove acentos e converte o texto para minúsculas."""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )


# ============================================================
# ANÁLISE COM SPACY + CLASSES
# ============================================================

def analisar(glosas):
    """
    spaCy + tabela CLASSES:
    informa ao LLM o que foi soletrado e a classe de cada sinal.

    Sinais conhecidos usam a tabela CLASSES.
    Sinais fora da tabela podem ser analisados pelo spaCy.
    """

    soletradas = [g for g in glosas if g.isupper()]

    tipos = []
    verbos = []

    for g in glosas:

        if g in soletradas:
            continue

        classe = CLASSES.get(sem_acento(g))

        if classe is None:
            # Fora do vocabulário: consulta o spaCy.
            doc = nlp(g.lower())

            if (
                doc
                and doc[0].pos_ == "VERB"
                and g.lower().endswith(("ar", "er", "ir", "or"))
            ):
                classe = "verbo"

        if classe:
            tipos.append(g + " = " + classe)

        if classe == "verbo":
            verbos.append(g)

    dicas = []

    if soletradas:
        dicas.append(
            "Soletradas: " + ", ".join(soletradas) + "."
        )

    if tipos:
        dicas.append(
            "Classes: " + "; ".join(tipos) + "."
        )

    if not verbos:
        dicas.append(
            "Nenhum verbo entre os sinais: "
            "infira o mais provavel (ter, ser, estar, ir...)."
        )

    return " ".join(dicas)


# ============================================================
# VALIDAÇÃO DA FRASE
# ============================================================

def confere(glosas, frase):
    """
    Verifica se todas as palavras dos sinais aparecem na frase.

    Para palavras comuns, considera o lema ou radical.
    Para palavras soletradas, verifica o texto diretamente.
    """

    doc = nlp(frase)

    formas = {
        sem_acento(t.text)
        for t in doc
    } | {
        sem_acento(t.lemma_)
        for t in doc
    }

    texto = sem_acento(frase)

    for g in glosas:

        soletrada = g.isupper()
        g = sem_acento(g)

        if soletrada:
            if g not in texto:
                return False

            continue

        radical = g[:max(4, len(g) - 3)]

        if (
            g not in formas
            and not any(f.startswith(radical) for f in formas)
        ):
            return False

    return True


def sujeito_invalido(frase):
    """
    Analisa a estrutura sintática com spaCy.

    Quem sente medo ou vontade precisa ser uma pessoa.
    Retorna o motivo do erro ou uma string vazia.
    """

    for t in nlp(frase):

        if (
            sem_acento(t.lemma_) in SENTIMENTOS
            or sem_acento(t.text) in SENTIMENTOS
        ):

            for filho in t.head.children:

                if filho.dep_ == "nsubj":

                    classe = CLASSES.get(
                        sem_acento(filho.lemma_),
                        "pessoa"
                    )

                    if not classe.startswith("pessoa"):

                        return (
                            "Quem sente '"
                            + t.text
                            + "' deve ser uma pessoa, e '"
                            + filho.text
                            + "' nao e. Corrija o sujeito."
                        )

    return ""


# ============================================================
# INTEGRAÇÃO COM GROQ (MODELO PRINCIPAL)
# ============================================================


def chamar_groq(prompt):
    """
    Envia o prompt para a API Groq.
    O Groq é o modelo principal do tradutor.
    """

    chave = os.environ.get("GROQ_API_KEY")

    if not chave:
        raise RuntimeError(
            "GROQ_API_KEY não foi definida no ambiente."
        )

    try:
        from groq import Groq

    except ImportError:
        raise RuntimeError(
            "Biblioteca Groq não instalada. "
            "Execute: python -m pip install groq"
        ) from None

    cliente = Groq(
        api_key=chave,
        timeout=20,
        max_retries=1
    )

    resposta = cliente.chat.completions.create(
        model=GROQ_MODELO,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um tradutor de glosas de Libras "
                    "para português brasileiro. "
                    "Siga todas as instruções recebidas. "
                    "Retorne somente a frase traduzida."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,

        # Mais espaço para o raciocínio e a frase final.
        max_completion_tokens=1024,

        # Reduz o esforço de raciocínio para uma tarefa curta.
        reasoning_effort="low"
    )

    escolha = resposta.choices[0]
    texto = escolha.message.content

    if not isinstance(texto, str) or not texto.strip():
        motivo = escolha.finish_reason

        raise RuntimeError(
            "O Groq retornou uma resposta vazia. "
            "Motivo de encerramento: "
            + str(motivo)
        )

    return texto.strip()



# ============================================================
# INTEGRAÇÃO COM GEMINI (MODELO RESERVA)
# ============================================================

def _chamar_gemini(prompt, modelo):
    """
    Envia o prompt para a API Gemini.

    O Gemini é utilizado quando o Groq falha.
    """

    chave = os.environ.get("GEMINI_API_KEY")

    if not chave:
        raise RuntimeError(
            "GEMINI_API_KEY não foi definida no ambiente."
        )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + modelo
        + ":generateContent"
    )

    corpo = json.dumps({
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0
        }
    }).encode()

    pedido = urllib.request.Request(
        url,
        data=corpo,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": chave
        }
    )

    dados = None

    for tentativa in range(3):

        try:

            with urllib.request.urlopen(
                pedido,
                timeout=12
            ) as resposta:

                dados = json.load(resposta)

            break

        except urllib.error.HTTPError as erro:

            if erro.code in (429, 500, 503) and tentativa < 2:

                time.sleep(1 + tentativa)
                continue

            raise RuntimeError(
                "Gemini "
                + modelo
                + " "
                + str(erro.code)
                + ": "
                + erro.read().decode(errors="ignore")[:300]
            ) from None

        except (TimeoutError, urllib.error.URLError) as erro:

            if tentativa < 2:
                time.sleep(1 + tentativa)
                continue

            raise RuntimeError(
                "Gemini "
                + modelo
                + " sem resposta: "
                + str(erro)
            ) from None

    try:

        partes = dados["candidates"][0]["content"]["parts"]

    except (KeyError, IndexError, TypeError):

        raise RuntimeError(
            "A API do Gemini não retornou texto. Resposta: "
            + str(dados)[:300]
        ) from None

    return "".join(
        p.get("text", "")
        for p in partes
    ).strip()


# ============================================================
# GERENCIAMENTO DOS MODELOS
# ============================================================

def chamar_llm(prompt):
    """
    Ordem de prioridade:

    1. Groq (modelo principal).
    2. Gemini principal.
    3. Modelos Gemini adicionais configurados como reserva.

    Se todos falharem, informa o último erro.
    """

    ultimo_erro = None

    # --------------------------------------------------------
    # PRIMEIRA TENTATIVA: GROQ
    # --------------------------------------------------------

    if os.environ.get("GROQ_API_KEY"):

        try:

            resultado = chamar_groq(prompt)

            if resultado:
                print("LLM utilizado: Groq -", GROQ_MODELO)
                return resultado

            raise RuntimeError(
                "Groq retornou uma resposta vazia."
            )

        except Exception as erro:

            ultimo_erro = erro

            print(
                "Groq falhou:",
                str(erro)[:200]
            )

    else:

        print(
            "GROQ_API_KEY não configurada. "
            "Pulando para o Gemini."
        )

    # --------------------------------------------------------
    # SEGUNDA TENTATIVA: GEMINI
    # --------------------------------------------------------

    if os.environ.get("GEMINI_API_KEY"):

        modelos = [GEMINI_MODELO] + GEMINI_RESERVA

        for modelo in modelos:

            try:

                resultado = _chamar_gemini(
                    prompt,
                    modelo
                )

                if resultado:

                    print(
                        "LLM utilizado: Gemini -",
                        modelo
                    )

                    return resultado

                raise RuntimeError(
                    "Gemini retornou uma resposta vazia."
                )

            except Exception as erro:

                ultimo_erro = erro

                print(
                    "Gemini falhou:",
                    modelo,
                    str(erro)[:200]
                )

    else:

        print(
            "GEMINI_API_KEY não configurada."
        )

    # --------------------------------------------------------
    # NENHUM MODELO FUNCIONOU
    # --------------------------------------------------------

    if ultimo_erro:
        raise RuntimeError(
            "Groq e Gemini indisponíveis. Último erro: "
            + str(ultimo_erro)
        )

    raise RuntimeError(
        "Nenhuma chave configurada. "
        "Defina GROQ_API_KEY ou GEMINI_API_KEY."
    )


# ============================================================
# LIMPEZA DA RESPOSTA
# ============================================================

def limpar(texto):
    """
    Mantém somente a frase.

    Remove rótulos, aspas e asteriscos que alguns modelos
    podem acrescentar na resposta.
    """

    linhas = [
        l.strip()
        for l in texto.splitlines()
        if l.strip()
    ]

    frase = linhas[0] if linhas else ""

    if frase.lower().startswith("frase:"):
        frase = frase[6:]

    return frase.strip(" *\"'")


# ============================================================
# VERIFICAÇÃO DE PROBLEMAS NA TRADUÇÃO
# ============================================================

def problema(glosas, frase):
    """
    Identifica se faltou alguma palavra ou se o sujeito
    da frase não faz sentido segundo as regras do projeto.
    """

    if not confere(
        [g for g in glosas if not g.isupper()],
        frase
    ):

        return (
            "Faltou alguma palavra. Use TODAS: "
            + " ".join(glosas)
            + "."
        )

    return sujeito_invalido(frase)


# ============================================================
# FUNÇÃO PRINCIPAL DO TRADUTOR
# ============================================================

def montar_frase(glosas):
    """
    Devolve:

    (frase, confere, usou_llm)

    Prioridade:
    1. Groq.
    2. Gemini.
    3. Fallback local.

    Mantém o mesmo contrato de retorno utilizado pelo app.py.
    """

    glosas = [
        g for g in glosas
        if g
    ]

    if not glosas:
        return "", True, False

    chave = tuple(glosas)

    # --------------------------------------------------------
    # VERIFICA CACHE
    # --------------------------------------------------------

    if chave in CACHE:
        return CACHE[chave]

    print("Glosas recebidas:", glosas)

    # --------------------------------------------------------
    # MONTA PROMPT COM AS INFORMAÇÕES DO SPACY + CLASSES
    # --------------------------------------------------------

    prompt = PROMPT.format(
        glosas=" ".join(glosas),
        analise=analisar(glosas)
    )

    # --------------------------------------------------------
    # TENTA GERAR A FRASE COM GROQ OU GEMINI
    # --------------------------------------------------------

    try:

        frase = limpar(
            chamar_llm(prompt)
        )

        motivo = problema(glosas, frase) if frase else ""

        # ----------------------------------------------------
        # SEGUNDA TENTATIVA SE A FRASE PRECISAR DE CORREÇÃO
        # ----------------------------------------------------

        if motivo:

            print(
                "Corrigindo:",
                frase,
                "->",
                motivo
            )

            try:

                nova = limpar(
                    chamar_llm(
                        prompt
                        + " "
                        + frase
                        + "\n("
                        + motivo
                        + ")\nFrase:"
                    )
                )

                if nova:
                    frase = nova

            except Exception as erro:

                # A primeira frase continua valendo.
                print(
                    "Segunda tentativa falhou:",
                    str(erro)[:150]
                )

        # ----------------------------------------------------
        # VERIFICA SE A RESPOSTA ESTÁ VAZIA
        # ----------------------------------------------------

        if not frase:

            raise RuntimeError(
                "O LLM devolveu uma resposta vazia."
            )

        # ----------------------------------------------------
        # VALIDA A FRASE FINAL
        # ----------------------------------------------------

        resultado = (
            frase,
            confere(glosas, frase) and not sujeito_invalido(frase),
            True
        )

        # Somente resultados válidos entram no cache.
        if resultado[1]:
            CACHE[chave] = resultado

        return resultado

    # --------------------------------------------------------
    # FALLBACK LOCAL
    # --------------------------------------------------------

    except Exception as erro:

        print(
            "Groq e Gemini indisponíveis:",
            erro
        )

        return (
            " ".join(glosas).capitalize() + ".",
            False,
            False
        )


# ============================================================
# TESTES DIRETOS DO ARQUIVO
# ============================================================

if __name__ == "__main__":

    print(
        "Arquivo carregado:",
        os.path.abspath(__file__)
    )

    print(
        "Modelo Groq principal:",
        GROQ_MODELO
    )

    print(
        "Chave Groq configurada:",
        bool(os.environ.get("GROQ_API_KEY"))
    )

    print(
        "Modelo Gemini reserva:",
        GEMINI_MODELO
    )

    print(
        "Modelos Gemini adicionais:",
        GEMINI_RESERVA or "nenhum"
    )

    print(
        "Chave Gemini configurada:",
        bool(os.environ.get("GEMINI_API_KEY"))
    )

    testes = [
        ["filho", "medo", "conhecer", "américa"],
        ["américa", "medo", "filho"],
        ["vacina", "medo", "filho"],
        ["MARIA", "filho", "medo"]
    ]

    for sinais in testes:

        resultado, ok, usou_llm = montar_frase(sinais)

        print("\nSinais:", sinais)
        print("Frase:", resultado)
        print("Confere:", ok)
        print("LLM:", usou_llm)
