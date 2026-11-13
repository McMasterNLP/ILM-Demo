from tests.conftest import learner

PREFIX = "/api/v1"

def _put(client, paragraph: int, learner_id: str = "tester"):
    return client.put(
        f"{PREFIX}/me/progress/yusuf",
        json={"paragraph": paragraph},
        headers={"X-Learner-Id": learner_id},
    )

def test_progress_only_moves_forward(client, yusuf):
    _put(client, 2)
    body = _put(client, 0).json()

    assert body["paragraph"] == 2

def test_last_paragraph_completes_the_story(client, yusuf):
    assert _put(client, 1).json()["completed"] is False
    assert _put(client, 2).json()["completed"] is True

def test_paragraph_index_is_clamped(client, yusuf):
    body = _put(client, 99).json()

    assert body["paragraph"] == 2
    assert body["completed"] is True

def test_restart_clears_reading_but_keeps_the_quiz_score(client, yusuf):
    _put(client, 2)
    attempt = client.post(f"{PREFIX}/quiz/yusuf/attempts", headers=learner(client)).json()
    client.post(
        f"{PREFIX}/attempts/{attempt['id']}/answers",
        json={"question_id": "q1", "option_id": "o1"},
        headers=learner(client),
    )
    client.post(f"{PREFIX}/attempts/{attempt['id']}/finish", headers=learner(client))

    body = client.delete(f"{PREFIX}/me/progress/yusuf", headers=learner(client)).json()

    assert body["paragraph"] == 0
    assert body["completed"] is False
    assert body["quiz"] == {"correct": 1, "total": 1}

def test_summary_counts_reading_and_answers(client, yusuf):
    _put(client, 2)
    attempt = client.post(f"{PREFIX}/quiz/yusuf/attempts", headers=learner(client)).json()
    client.post(
        f"{PREFIX}/attempts/{attempt['id']}/answers",
        json={"question_id": "q1", "option_id": "o1"},
        headers=learner(client),
    )
    client.post(f"{PREFIX}/attempts/{attempt['id']}/finish", headers=learner(client))

    body = client.get(f"{PREFIX}/me/progress", headers=learner(client)).json()

    assert body["stories_started"] == 1
    assert body["stories_completed"] == 1
    assert body["questions_answered"] == 1

def test_progress_is_scoped_to_the_learner(client, yusuf):
    _put(client, 2, learner_id="one")

    body = client.get(f"{PREFIX}/me/progress", headers={"X-Learner-Id": "two"}).json()
    assert body["stories_started"] == 0
