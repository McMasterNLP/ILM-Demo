from tests.conftest import learner

PREFIX = "/api/v1"

def _finish_reading(client, story_id="yusuf", last=2):
    return client.put(
        f"{PREFIX}/me/progress/{story_id}",
        json={"paragraph": last},
        headers=learner(client),
    )

def test_quiz_locked_before_story_is_finished(client, yusuf):
    response = client.get(f"{PREFIX}/quiz/yusuf", headers=learner(client))
    assert response.status_code == 403
    assert "Finish reading" in response.json()["detail"]

def test_reading_the_map_does_not_unlock_the_quiz(client, yusuf):
    assert client.get(f"{PREFIX}/stories/yusuf/graph").status_code == 200

    response = client.get(f"{PREFIX}/quiz/yusuf", headers=learner(client))
    assert response.status_code == 403

def test_quiz_opens_once_the_story_is_finished(client, yusuf):
    _finish_reading(client)

    response = client.get(f"{PREFIX}/quiz/yusuf", headers=learner(client))
    assert response.status_code == 200
    assert len(response.json()["questions"]) == 1

def test_attempt_cannot_be_started_while_locked(client, yusuf):
    response = client.post(f"{PREFIX}/quiz/yusuf/attempts", headers=learner(client))
    assert response.status_code == 403

def test_unlock_after_blocks_until_enough_stories_are_done(client, yusuf, session):
    from app.models import Story, StoryTranslation

    session.add(
        Story(id="salih", level="advanced", minutes=14, unlock_after=2, published=True)
    )
    session.add(
        StoryTranslation(
            story_id="salih", lang="en", title="Prophet Salih (AS)", status="approved"
        )
    )
    session.flush()

    _finish_reading(client)

    response = client.get(f"{PREFIX}/quiz/salih", headers=learner(client))
    assert response.status_code == 403
    assert response.json()["detail"] == "Story is locked"

def test_gate_is_per_learner(client, yusuf):
    _finish_reading(client)

    other = {"X-Learner-Id": "someone-else"}
    assert client.get(f"{PREFIX}/quiz/yusuf", headers=other).status_code == 403
