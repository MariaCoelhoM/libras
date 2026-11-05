import json
import os
import unicodedata
import urllib.error
import urllib.request
import spacy

# A chave deve ser definida fora do código, na variável de ambiente GEMINI_API_KEY.
GEMINI_MODELO = os.environ.get("GEMINI_MODELO", "gemini-3.8-flash").strip()
MODELO_CLAUDE = "claude-haiku-5-5"

nlp = spacy.load("pt_core_news_sm")   # python -m spacy download pt_core_news_sm

PROMPT = """Voce converte sinais de Libras, na ordem em que foram feitos, em uma frase curta e natural em portugues.
Regras:
- Use TODAS as palavras, sem mudar o sentido e sem inventar informacao nova.
- Pode acrescentar verbos (ter, ser, estar, ir...), artigos e preposicoes, e flexionar os verbos.
- Palavras em MAIUSCULAS foram soletradas no alfabeto manual (nomes ou siglas): mantenha-as, com inicial maiuscula.
- Responda so com a frase, terminada em ponto final (or em ? se for pergunta).

Exemplos:
Sinais: aluno medo conhecer américa
Frase: O aluno tem medo de conhecer a América.

Sinais: filho vontade banheiro
Frase: O filho tem vontade de ir ao banheiro.

Sinais: maçã amarelo
Frase: A maçã é amarela.

Sinais: banco esquina
Frase: O banco fica na esquina.

Sinais: {glosas}
Analise (spaCy): {analise}
Frase:"""


def sem_acento(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")


def analisar(glosas):
    """spaCy: diz ao LLM o que foi soletrado e quais sinais sao verbos (no infinitivo, como em Libras).
    A classe gramatical de uma palavra isolada erra as vezes (ex.: 'américa' vira verbo), por isso so
    conta como verbo o que o spaCy marca como VERB e termina como infinitivo."""
    soletradas = [g for g in glosas if g.isupper()]
    verbos = [g for g in glosas
              if g not in soletradas and nlp(g.lower())[0].pos_ == "VERB" and g.lower().endswith(("ar", "er", "ir", "or"))]
    dicas = []
    if soletradas:
        dicas.append("Soletradas: " + ", ".join(soletradas) + ".")
    if verbos:
        dicas.append("Verbos: " + ", ".join(verbos) + ".")
    else:
        dicas.append("Nenhum verbo entre os sinais: infira o mais provavel (ter, ser, estar, ir...).")
    return " ".join(dicas)


def confere(glosas, frase):
    """Todas as palavras dos sinais aparecem na frase gerada? Compara pelo lema (spaCy) ou pelo radical."""
    doc = nlp(frase)
    formas = {sem_acento(t.text) for t in doc} | {sem_acento(t.lemma_) for t in doc}
    for g in glosas:
        g = sem_acento(g)
        radical = g[: max(4, len(g) - 3)]
        if g not in formas and not any(f.startswith(radical) for f in formas):
            return False
    return True


def chamar_gemini(prompt):
    chave = os.environ.get("GEMINI_API_KEY")
    if not chave:
        raise RuntimeError("GEMINI_API_KEY não foi definida no ambiente.")
    url = "https://generativelanguage.googleapis.com/v1beta/models/" + GEMINI_MODELO + ":generateContent"
    corpo = json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}}).encode()
    pedido = urllib.request.Request(url, data=corpo, headers={
        "Content-Type": "application/json", "x-goog-api-key": chave})
    try:
        with urllib.request.urlopen(pedido, timeout=20) as resposta:
            dados = json.load(resposta)
    except urllib.error.HTTPError as erro:  # mostra o motivo que o Google devolveu (chave invalida, limite, modelo...)
        raise RuntimeError("Gemini " + str(erro.code) + ": " + erro.read().decode(errors="ignore")[:300]) from None
    try:
        partes = dados["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("A API do Gemini não retornou texto. Resposta: " + str(dados)[:300]) from None
    return "".join(p.get("text", "") for p in partes).strip()


def chamar_claude(prompt):
    from anthropic import Anthropic  # chave em ANTHROPIC_API_KEY
    resposta = Anthropic(timeout=15, max_retries=1).messages.create(
        model=MODELO_CLAUDE, max_tokens=120, messages=[{"role": "user", "content": prompt}]
    )
    return resposta.content[0].text.strip()


def chamar_llm(prompt):
    if os.environ.get("GEMINI_API_KEY"):
        return chamar_gemini(prompt)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return chamar_claude(prompt)
    raise RuntimeError("nenhuma chave configurada: defina GEMINI_API_KEY (gratis) ou ANTHROPIC_API_KEY")


def limpar(texto):
    """Fica so com a frase: tira 'Frase:', aspas e asteriscos que alguns modelos acrescentam."""
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    frase = linhas[0] if linhas else ""
    if frase.lower().startswith("frase:"):
        frase = frase[6:]
    return frase.strip(" *\"'")


def montar_frase(glosas):
    """Devolve (frase, confere, usou_llm). Sem LLM, devolve as palavras na ordem dos sinais."""
    glosas = [g for g in glosas if g]
    if not glosas:
        return "", True, False
    prompt = PROMPT.format(glosas=" ".join(glosas), analise=analisar(glosas))
    try:
        frase = limpar(chamar_llm(prompt))
        if not confere(glosas, frase):  # o LLM esqueceu ou trocou uma palavra: tenta mais uma vez
            frase = limpar(chamar_llm(prompt + " " + frase + "\n(Faltou alguma palavra. Use TODAS: " + " ".join(glosas) + ".)\nFrase:"))
        if not frase:
            raise RuntimeError("o LLM devolveu uma resposta vazia")
        return frase, confere(glosas, frase), True
    except Exception as erro:
        print("LLM indisponivel:", erro)
        return " ".join(glosas).capitalize() + ".", False, False


if __name__ == "__main__":
    print("Arquivo carregado:", os.path.abspath(__file__))
    print("Modelo Gemini:", GEMINI_MODELO)
    print("Chave Gemini configurada:", bool(os.environ.get("GEMINI_API_KEY")))
    resultado, ok, usou_llm = montar_frase(["filho", "medo", "conhecer", "américa"])
    print("Frase Final:", resultado)
    print("Processado via LLM:", usou_llm)
    