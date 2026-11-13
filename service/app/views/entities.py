from fastapi import APIRouter, HTTPException

from app.deps import SessionDep
from app.repositories import content as content_repo
from app.schemas import EntityAppearanceOut, EntityDetailOut, EntityRelationOut

router = APIRouter()

@router.get("/{key}", response_model=EntityDetailOut)
def retrieve_entity(key: str, session: SessionDep):
    entity = content_repo.entity_by_key(session, key)
    if entity is None or entity.status != content_repo.APPROVED:
        raise HTTPException(404, "Entity not found")

    stories = content_repo.stories_featuring(session, entity)
    relations = content_repo.relations_for_entity(session, entity)

    edges: list[EntityRelationOut] = []
    for relation in relations:
        outgoing = relation.subject_id == entity.id
        other = relation.object if outgoing else relation.subject
        edges.append(
            EntityRelationOut(
                other_key=other.key,
                other_label=content_repo.localized(other.translations, "label"),
                label=content_repo.localized(relation.predicate.translations, "label"),
                outgoing=outgoing,
            )
        )

    return EntityDetailOut(
        key=entity.key,
        type=entity.type,
        label=content_repo.localized(entity.translations, "label"),
        category=content_repo.localized(entity.translations, "category"),
        description=content_repo.localized(entity.translations, "description"),
        appearances=[
            EntityAppearanceOut(
                story_id=story.id,
                title=content_repo.localized(story.translations, "title", only_approved=True),
            )
            for story in stories
        ],
        relations=edges,
    )
