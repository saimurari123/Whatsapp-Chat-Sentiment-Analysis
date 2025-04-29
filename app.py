def analyze_chat(uploaded_file, model, vectorizer):
    opinion = {}
    messages = []
    pos, neg = 0, 0

    for line in uploaded_file:
        try:
            line = line.decode('utf-8')
            name_msg = line.split('-')[1]
            name = name_msg.split(':')[0].strip()
            chat = name_msg.split(':', 1)[1].strip()
            messages.append((name, chat))

            if opinion.get(name) is None:
                opinion[name] = [0, 0]

            normalized = normalize_corpus([chat])
            vectorized = vectorizer.transform(normalized)
            res = model.predict(vectorized)[0]

            if res == 'positive':
                pos += 1
                opinion[name][0] += 1
            else:
                neg += 1
                opinion[name][1] += 1

        except Exception:
            continue

    return opinion, pos, neg, messages
