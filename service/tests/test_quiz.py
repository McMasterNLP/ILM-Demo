from tests.conftest import learner

PREFIX = "/api/v1"

def _unlock(client):
    client.put(
        f"{PREFIX}/me/progress/yusuf", json={"paragraph": 2}, headers=learner(client)
    )

def _start(client):
    return client.post(f"{PREFIX}/quiz/yusuf/attempts", headers=learner(client)).json()

def test_options_never_reveal_the_answer(client, yusuf):
    _unlock(client)

    payload = client.get(f"{PREFIX}/quiz/yusuf", headers=learner(client)).json()
    for question in payload["questions"]:
        for option in question["options"]:
            assert set(option) == {"id", "label"}

def test_correct_answer_scores(client, yusuf):
    _unlock(client)
    attempt = _start(client)

    response = client.post(
        f"{PREFIX}/attempts/{attempt['id']}/answers",
        json={"question_id": "q1", "option_id": "o1"},
        headers=learner(client),
    )
    body = response.json()

    assert body["is_correct"] is True
    assert body["score"] == 1
    assert body["follow_up"] is None

def test_wrong_answer_returns_edge_explanation_and_followup(client, yusuf):
    _unlock(client)
    attempt = _start(client)

    body = client.post(
        f"{PREFIX}/attempts/{attempt['id']}/answers",
        json={"question_id": "q1", "option_id": "o2"},
        headers=learner(client),
    ).json()

    assert body["is_correct"] is False
    assert body["correct_option_id"] == "o1"
    assert body["kg_edge"] == "Yusuf (AS) ⟶ (son of) ⟶ Yaqub (AS)"
    assert body["explanation"]["en"].startswith("Yusuf is the son")
    assert body["follow_up"]["id"] == "q1f"
    assert body["score"] == 0

def test_followup_teaches_but_does_not_score(client, yusuf):
    _unlock(client)
    attempt = _start(client)

    client.post(
        f"{PREFIX}/attempts/{attempt['id']}/answers",
        json={"question_id": "q1", "option_id": "o2"},
        headers=learner(client),
    )
    body = client.post(
        f"{PREFIX}/attempts/{attempt['id']}/answers",
        json={"question_id": "q1f", "option_id": "o3", "was_followup": True},
        headers=learner(client),
    ).json()

    assert body["is_correct"] is True
    assert body["score"] == 0

def test_answering_twice_does_not_double_score(client, yusuf):
    _unlock(client)
    attempt = _start(client)

    for _ in range(2):
        body = client.post(
            f"{PREFIX}/attempts/{attempt['id']}/answers",
            json={"question_id": "q1", "option_id": "o1"},
            headers=learner(client),
        ).json()

    assert body["score"] == 1

def test_option_from_another_question_is_rejected(client, yusuf):
    _unlock(client)
    attempt = _start(client)

    response = client.post(
        f"{PREFIX}/attempts/{attempt['id']}/answers",
        json={"question_id": "q1", "option_id": "o3"},
        headers=learner(client),
    )
    assert response.status_code == 400

def test_wrong_answer_records_concept_mastery(client, yusuf, session):
    from app.models import ConceptMastery

    _unlock(client)
    attempt = _start(client)
    client.post(
        f"{PREFIX}/attempts/{attempt['id']}/answers",
        json={"question_id": "q1", "option_id": "o2"},
        headers=learner(client),
    )

    row = session.get(ConceptMastery, {"user_id": "tester", "relation_id": "r1"})
    assert row is not None
    assert row.wrong_count == 1

def test_finished_attempt_rejects_more_answers(client, yusuf):
    _unlock(client)
    attempt = _start(client)
    client.post(f"{PREFIX}/attempts/{attempt['id']}/finish", headers=learner(client))

    response = client.post(
        f"{PREFIX}/attempts/{attempt['id']}/answers",
        json={"question_id": "q1", "option_id": "o1"},
        headers=learner(client),
    )
    assert response.status_code == 409

def test_attempt_belongs_to_its_learner(client, yusuf):
    _unlock(client)
    attempt = _start(client)

    response = client.post(
        f"{PREFIX}/attempts/{attempt['id']}/answers",
        json={"question_id": "q1", "option_id": "o1"},
        headers={"X-Learner-Id": "intruder"},
    )
    assert response.status_code == 404
