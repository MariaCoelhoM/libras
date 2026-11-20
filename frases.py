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

# Tentativas (o Groq às vezes devolve resposta vazia; o Gemini, 503).
GROQ_TENTATIVAS = 3
GEMINI_TENTATIVAS = 4

# Quantas frases novas pedir ao LLM quando a resposta é inválida
# (ex.: "forneça os sinais que deseja traduzir").
MAX_GERACOES = 3


# ============================================================
# CARREGAMENTO DO SPACY
# ============================================================

nlp = spacy.load("pt_core_news_sm")


# ============================================================
# CHAVES DE API
# ============================================================

def _chave(nome):
    """Lê a chave do ambiente sem espaços nem quebras de linha nas pontas."""
    return (os.environ.get(nome) or "").strip()


def _ocultar(texto):
    """Troca as chaves de API por *** (mensagens de erro podem conter a chave)."""
    for nome in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        c = _chave(nome)
        if c:
            texto = texto.replace(c, "***")
    return texto


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

    # V-Librasil (Núcleo Comum)
    "surdo": "pessoa",
    "compreender": "verbo",
    "onde": "interrogativo (a frase e uma pergunta)",
    "nao": "negacao",
    "sim": "afirmacao",
    "desculpa": "expressao de desculpa",
    "por favor": "expressao de cortesia",
}

SENTIMENTOS = {
    g for g, c in CLASSES.items()
    if c.startswith("sentimento")
}

# Sinais que podem aparecer sozinhos, como uma frase curta.
EXPRESSOES = {"sim", "nao", "desculpa", "por favor"}


# ============================================================
# PROMPT DE TRADUÇÃO
# ============================================================

# ATENCAO: os exemplos abaixo NAO podem estar em glosas_teste.csv.
# Prefira exemplos tirados de glosas_desenvolvimento.csv.
PROMPT = """Voce converte sinais de Libras, na ordem em que foram feitos, em uma frase curta e natural em portugues.

Regras:
- Use TODAS as palavras, sem mudar o sentido e sem inventar informacao nova.
- Pode acrescentar verbos (ter, ser, estar, ir...), artigos e preposicoes, e flexionar os verbos.
- Em Libras a ordem nao segue o portugues: o primeiro sinal costuma ser o assunto ou o objeto, nao necessariamente o sujeito.
- Sentimentos (medo, vontade) sao sentidos por pessoas, nunca por lugares, objetos ou animais. Escolha sempre o sujeito e o objeto mais plausiveis no dia a dia.
- Palavras em MAIUSCULAS foram soletradas no alfabeto manual (nomes ou siglas): mantenha-as exatamente como estao, apenas com inicial maiuscula.
- Os sinais estao sempre na linha "Sinais:". Nunca peca mais informacao: se houver so um sinal, escreva uma frase curta com ele.
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

Sinais: sapo amarelo
Frase: O sapo é amarelo.

Sinais: espelho banheiro
Frase: O espelho fica no banheiro.

Sinais: por favor onde esquina
Frase: Por favor, onde fica a esquina?

Sinais: espelho não ruim
Frase: O espelho não é ruim.

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
    comuns = []

    for g in glosas:

        if g in soletradas:
            continue

        comuns.append(sem_acento(g))

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

        if comuns and all(c in EXPRESSOES for c in comuns):
            dicas.append(
                "Expressao isolada: escreva-a como uma frase curta, "
                "sem acrescentar informacao."
            )

        else:
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
    Para palavras soletradas e sinais com mais de uma palavra
    (ex.: "por favor"), verifica o texto diretamente.
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

        if soletrada or " " in g:
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


# Trechos que mostram que o LLM nao traduziu (pediu os sinais, etc.).
MARCAS_META = (
    "forneca",
    "forneco",
    "sinais",
    "traduza",
    "traduzir",
    "que deseja",
    "entendido",
    "informe os",
    "envie os",
)


def resposta_invalida(glosas, frase):
    """
    True se a resposta nao e uma traducao: vazia, so pontuacao,
    muito longa ou pedindo os sinais ao usuario.
    """

    texto = sem_acento(frase)

    if not any(c.isalpha() for c in texto):
        return True

    if len(frase.split()) > 20:
        return True

    alvo = sem_acento(" ".join(glosas))

    return any(m in texto and m not in alvo for m in MARCAS_META)


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

class RespostaVazia(RuntimeError):
    """O Groq respondeu sem texto (o modelo de raciocinio gastou tudo pensando)."""


def chamar_groq(prompt, tentativa=0):
    """
    Envia o prompt para a API Groq.
    O Groq é o modelo principal do tradutor.

    `tentativa` (0, 1, 2...) so e usado nas repeticoes apos resposta vazia:
    repetir exatamente o mesmo pedido tende a dar a mesma resposta vazia,
    entao a temperatura sobe e, na ultima tentativa, o raciocinio aumenta.
    """

    chave = _chave("GROQ_API_KEY")

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
        temperature=(0.2, 0.5, 0.8)[min(tentativa, 2)],

        # Mais espaço para o raciocínio e a frase final.
        max_completion_tokens=1024,

        # Esforço baixo para uma tarefa curta; sobe na ultima repeticao.
        reasoning_effort="medium" if tentativa >= GROQ_TENTATIVAS - 1 else "low"
    )

    escolha = resposta.choices[0]
    texto = escolha.message.content

    if not isinstance(texto, str) or not texto.strip():
        uso = getattr(resposta, "usage", None)

        raise RespostaVazia(
            "O Groq retornou uma resposta vazia. "
            "Motivo de encerramento: "
            + str(escolha.finish_reason)
            + ". Tokens de saida: "
            + str(getattr(uso, "completion_tokens", "?"))
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

    chave = _chave("GEMINI_API_KEY")

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

    for tentativa in range(GEMINI_TENTATIVAS):

        ultima = tentativa == GEMINI_TENTATIVAS - 1

        try:

            with urllib.request.urlopen(
                pedido,
                timeout=15
            ) as resposta:

                dados = json.load(resposta)

            break

        except urllib.error.HTTPError as erro:

            detalhe = erro.read().decode(errors="ignore")

            # 429 por COTA esgotada nao passa esperando segundos: falha logo.
            sem_cota = erro.code == 429 and "quota" in detalhe.lower()

            if erro.code in (429, 500, 503) and not ultima and not sem_cota:

                # Espera 2, 4, 8 s entre as tentativas.
                time.sleep(2 * (2 ** tentativa))
                continue

            raise RuntimeError(
                "Gemini "
                + modelo
                + " "
                + str(erro.code)
                + ": "
                + detalhe[:300]
            ) from None

        except (TimeoutError, urllib.error.URLError) as erro:

            if not ultima:
                time.sleep(2 * (2 ** tentativa))
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

    1. Groq (modelo principal), repetindo se vier resposta vazia.
    2. Gemini principal.
    3. Modelos Gemini adicionais configurados como reserva.

    Se todos falharem, informa o último erro.
    """

    ultimo_erro = None

    # --------------------------------------------------------
    # PRIMEIRA TENTATIVA: GROQ
    # --------------------------------------------------------

    if _chave("GROQ_API_KEY"):

        for tentativa in range(GROQ_TENTATIVAS):

            try:

                resultado = chamar_groq(prompt, tentativa)

                print("LLM utilizado: Groq -", GROQ_MODELO)
                return resultado

            except RespostaVazia as erro:

                ultimo_erro = erro

                print(
                    "Groq falhou (tentativa "
                    + str(tentativa + 1) + "/" + str(GROQ_TENTATIVAS)
                    + "):",
                    _ocultar(str(erro))[:200]
                )

                if tentativa < GROQ_TENTATIVAS - 1:
                    time.sleep(1 + tentativa)

            except Exception as erro:

                ultimo_erro = erro

                print(
                    "Groq falhou:",
                    _ocultar(str(erro))[:200]
                )

                break

    else:

        print(
            "GROQ_API_KEY não configurada. "
            "Pulando para o Gemini."
        )

    # --------------------------------------------------------
    # SEGUNDA TENTATIVA: GEMINI
    # --------------------------------------------------------

    if _chave("GEMINI_API_KEY"):

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
                    _ocultar(str(erro))[:200]
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
            + _ocultar(str(ultimo_erro))
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

    # Alguns modelos repetem o enunciado ("Sinais: ...") antes da frase.
    # Se houver uma linha "Frase: ...", ela e a resposta.
    for l in linhas:
        if l.lower().startswith("frase:"):
            conteudo = l[6:].strip(" *\"'")
            if conteudo:
                return conteudo

    # Senao, descarta as linhas que apenas repetem o enunciado.
    linhas = [
        l for l in linhas
        if not sem_acento(l).startswith(("sinais:", "analise:", "frase:"))
    ]

    frase = linhas[0] if linhas else ""

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
    3. Fallback local (usou_llm = False).

    Respostas inválidas do LLM (pedir os sinais, só "?", etc.)
    nunca contam como saída do LLM: o LLM é chamado de novo e,
    se continuar inválido, a frase sai pelo fallback local.

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

        frase = ""

        for geracao in range(MAX_GERACOES):

            candidata = limpar(chamar_llm(prompt))

            # Resposta que nao e traducao: pede outra do zero.
            if resposta_invalida(glosas, candidata):

                print(
                    "Resposta invalida do LLM ("
                    + str(geracao + 1) + "/" + str(MAX_GERACOES) + "):",
                    candidata[:80]
                )

                continue

            frase = candidata
            motivo = problema(glosas, frase)

            if not motivo:
                break

            # ------------------------------------------------
            # SEGUNDA TENTATIVA SE A FRASE PRECISAR DE CORREÇÃO
            # ------------------------------------------------

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

                if nova and not resposta_invalida(glosas, nova):
                    frase = nova

            except Exception as erro:

                # A primeira frase continua valendo.
                print(
                    "Segunda tentativa falhou:",
                    _ocultar(str(erro))[:150]
                )

            break

        # ----------------------------------------------------
        # NENHUMA RESPOSTA VALIDA
        # ----------------------------------------------------

        if not frase:

            raise RuntimeError(
                "O LLM so devolveu respostas invalidas."
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
            "Sem frase do LLM, usando fallback local:",
            _ocultar(str(erro))
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
        bool(_chave("GROQ_API_KEY"))
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
        bool(_chave("GEMINI_API_KEY"))
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
