from __future__ import annotations

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.repositories.authoring import index_story_chunks
from app.models import (
    Entity,
    StoryEntity,
    StoryRelation,
    Source,
    EntityMention,
    EntityTranslation,
    Option,
    OptionTranslation,
    Paragraph,
    ParagraphTranslation,
    Predicate,
    PredicateTranslation,
    Question,
    QuestionTranslation,
    Relation,
    Story,
    StoryTranslation,
)

STORIES = [
    ("yusuf", "intermediate", 7, 12, None,
     "Prophet Yusuf (AS)", "Yusuf's journey from jealousy and betrayal to honor in Egypt.",
     "النبي يوسف عليه السلام", "رحلة يوسف من الحسد والخيانة إلى المكانة في مصر."),
    ("nuh", "beginner", 4, 8, None,
     "Prophet Nuh (AS)", "A story of faith and patience through the great flood.",
     "النبي نوح عليه السلام", "قصة إيمان وصبر عبر الطوفان العظيم."),
    ("musa", "advanced", 9, 18, None,
     "Prophet Musa (AS)", "From a baby on the Nile to a prophet facing Pharaoh.",
     "النبي موسى عليه السلام", "من رضيع في النيل إلى نبي يواجه فرعون."),
    ("yunus", "beginner", 3, 6, None,
     "Prophet Yunus (AS)", "A story of repentance and mercy inside the whale.",
     "النبي يونس عليه السلام", "قصة توبة ورحمة في بطن الحوت."),
    ("ibrahim", "intermediate", 5, 10, None,
     "Prophet Ibrahim (AS)", "A story of monotheism, sacrifice, and trial.",
     "النبي إبراهيم عليه السلام", "قصة توحيد وتضحية وابتلاء."),
    ("salih", "advanced", 6, 14, 2,
     "Prophet Salih (AS)", "The story of the people of Thamud, the she-camel, and the lesson within.",
     "النبي صالح عليه السلام", "قصة قوم ثمود والناقة والعبرة فيها."),
]

HEADINGS = {
    "yusuf": ("The Story of Prophet Yusuf (AS)", "قصة النبي يوسف عليه السلام"),
}

YUSUF_PARAGRAPHS = [
    ("Yusuf (AS) was the youngest son of Prophet Yaqub (AS), and his father loved him dearly — which stirred jealousy among his brothers.",
     "كان يوسف عليه السلام أصغر أبناء النبي يعقوب عليه السلام، وكان أبوه يحبه حبًا شديدًا، فأثار ذلك الحسد في نفوس إخوته.",
     ["yusuf", "yaqub", "brothers"]),
    ("Yusuf saw a strange dream and told it to his father. Yaqub warned him not to tell his brothers, fearing it would deepen their envy.",
     "رأى يوسف رؤيا عجيبة فقصّها على أبيه، فحذّره يعقوب من أن يخبر بها إخوته خشية أن يزداد حسدهم.",
     ["yusuf", "yaqub", "brothers"]),
    ("The brothers decided to get rid of him. They threw him into an abandoned well far from his father, then told Yaqub a false story about his disappearance.",
     "عزم الإخوة على التخلص منه، فألقوه في بئر مهجورة بعيدًا عن أبيه، ثم أخبروا يعقوب بقصة كاذبة عن اختفائه.",
     ["yusuf", "brothers", "well"]),
    (None, None, ["yusuf", "egypt"]),
    (None, None, ["yusuf", "alaziz"]),
    (None, None, ["yusuf", "zulaikha"]),
    (None, None, ["yusuf", "alaziz"]),
    (None, None, ["yusuf", "brothers"]),
    (None, None, ["yusuf", "yaqub"]),
]

PREDICATES = [
    ("child_of", False, "child of", "Child of {object}", "ابن / ابنة", "ابن / ابنة {object}"),
    ("father_of", False, "father of", "Father of {object}", "والد", "والد {object}"),
    ("sibling_of", True, "sibling of", "Sibling of {object}", "أخ / أخت", "أخ / أخت {object}"),
    ("sent_to", False, "sent to", "Sent to {object}", "أُرسل إلى", "أُرسل إلى {object}"),
    ("served_under", False, "served under", "Served under {object}", "خدم عند", "خدم عند {object}"),
    ("wife", False, "wife", "Husband of {object}", "زوجة", "زوج {object}"),
    ("thrown_into", False, "thrown into", "Thrown into {object}", "أُلقي في", "أُلقي في {object}"),
    ("moved_to", False, "moved to", "Lived in {object}", "انتقل إلى", "عاش في {object}"),
]

YUSUF_ENTITIES = [
    ("ibrahim", "person", 600, 62,
     ("Ibrahim (AS)", "Person — Prophet",
      "A prophet at the head of this family line, and the great-grandfather of Yusuf (AS)."),
     ("إبراهيم عليه السلام", "شخصية — نبي",
      "نبي في أعلى سلسلة هذه العائلة، وجدّ يوسف عليه السلام الأعلى.")),
    ("ishaq", "person", 600, 172,
     ("Ishaq (AS)", "Person — Prophet",
      "Son of Prophet Ibrahim (AS) and the father of Prophet Yaqub (AS)."),
     ("إسحاق عليه السلام", "شخصية — نبي",
      "ابن النبي إبراهيم عليه السلام ووالد النبي يعقوب عليه السلام.")),
    ("yaqub", "person", 600, 282,
     ("Yaqub (AS)", "Person — Prophet",
      "Father of Yusuf (AS). He loved his son dearly and warned him to keep his dream private."),
     ("يعقوب عليه السلام", "شخصية — نبي",
      "والد يوسف عليه السلام، أحبّه حبًا شديدًا وحذّره من أن يقصّ رؤياه.")),
    ("brothers", "person", 250, 352,
     ("Yusuf's Brothers", "Person — Group",
      "The older brothers of Yusuf (AS), whose envy set the events of the story in motion."),
     ("إخوة يوسف", "شخصية — مجموعة",
      "إخوة يوسف عليه السلام الأكبر سنًا، وحسدهم هو ما حرّك أحداث القصة.")),
    ("alaziz", "person", 952, 352,
     ("Al-Aziz (Minister of Egypt)", "Person — Official",
      "The minister in Egypt in whose household Yusuf (AS) lived and served."),
     ("العزيز (وزير مصر)", "شخصية — مسؤول",
      "وزير مصر الذي عاش يوسف عليه السلام في بيته وخدم عنده.")),
    ("yusuf", "person", 600, 442,
     ("Yusuf (AS)", "Person — Prophet",
      "Son of Prophet Yaqub, known for his truthful dreams and honesty. He faced jealousy and trials before rising to a position of authority in Egypt."),
     ("يوسف عليه السلام", "شخصية — نبي",
      "ابن النبي يعقوب، عُرف بصدق رؤاه وأمانته. واجه الحسد والابتلاء قبل أن يبلغ مكانة في مصر.")),
    ("zulaikha", "person", 1052, 552,
     ("Zulaikha (Al-Aziz's wife)", "Person",
      "The wife of Al-Aziz, part of the household Yusuf (AS) lived in."),
     ("زليخا (زوجة العزيز)", "شخصية",
      "زوجة العزيز، من أهل البيت الذي عاش فيه يوسف عليه السلام.")),
    ("well", "place", 372, 562,
     ("The Well", "Place",
      "The abandoned well far from his father, where Yusuf (AS) was thrown by his brothers."),
     ("البئر", "مكان",
      "البئر المهجورة البعيدة عن أبيه، حيث ألقاه إخوته.")),
    ("egypt", "place", 828, 562,
     ("Egypt", "Place",
      "The land Yusuf (AS) was taken to, and where he later rose to a position of authority."),
     ("مصر", "مكان",
      "الأرض التي نُقل إليها يوسف عليه السلام، وفيها بلغ لاحقًا مكانة ومنزلة.")),
]

YUSUF_RELATIONS = [
    ("r_ibrahim_ishaq", "ibrahim", "father_of", "ishaq"),
    ("r_ishaq_yaqub", "ishaq", "father_of", "yaqub"),
    ("r_yusuf_yaqub", "yusuf", "child_of", "yaqub"),
    ("r_yusuf_brothers", "yusuf", "sibling_of", "brothers"),
    ("r_yusuf_alaziz", "yusuf", "served_under", "alaziz"),
    ("r_alaziz_zulaikha", "alaziz", "wife", "zulaikha"),
    ("r_yusuf_well", "yusuf", "thrown_into", "well"),
    ("r_yusuf_egypt", "yusuf", "moved_to", "egypt"),
]

YUSUF_QUESTIONS = [
    ("q1", "relationship", "r_yusuf_yaqub", None,
     "What is the relationship between Yusuf (AS) and Yaqub (AS)?",
     "ما العلاقة بين يوسف ويعقوب عليهما السلام؟",
     "Yusuf (AS) is the son of Prophet Yaqub (AS). This edge sits at the centre of the story map.",
     "يوسف عليه السلام هو ابن النبي يعقوب عليه السلام، وهذه العلاقة في قلب خريطة القصة.",
     [("Son and father", "ابن ووالد", True), ("Brothers", "إخوة", False),
      ("Cousins", "أبناء عم", False), ("Teacher and student", "معلم وتلميذ", False)],
     ("Yaqub (AS) is Yusuf's …?", "يعقوب عليه السلام هو ... ليوسف؟",
      [("Father", "والد", True), ("Brother", "أخ", False), ("Uncle", "عم", False)])),
    ("q2", "relationship", "r_yusuf_yaqub", None,
     "Who is the father of Prophet Yusuf (AS)?",
     "من هو والد يوسف عليه السلام؟",
     "Yusuf (AS) is the son of Prophet Yaqub (AS), and the grandson of Prophet Ishaq (AS). This relationship appears in the story map you explored earlier.",
     "يوسف عليه السلام هو ابن النبي يعقوب عليه السلام وحفيد النبي إسحاق عليه السلام، وهذه العلاقة ظهرت في خريطة القصة التي استكشفتها.",
     [("Prophet Yaqub (AS)", "النبي يعقوب عليه السلام", True),
      ("Prophet Ishaq (AS)", "النبي إسحاق عليه السلام", False),
      ("Prophet Ibrahim (AS)", "النبي إبراهيم عليه السلام", False),
      ("Al-Aziz", "العزيز", False)],
     ("Yusuf (AS) is the son of which prophet?", "يوسف عليه السلام هو ابن أي نبي؟",
      [("Prophet Yaqub (AS)", "النبي يعقوب عليه السلام", True),
       ("Prophet Ismail (AS)", "النبي إسماعيل عليه السلام", False),
       ("Prophet Lut (AS)", "النبي لوط عليه السلام", False)])),
    ("q3", "identification", "r_yusuf_alaziz", None,
     "Who did Yusuf (AS) serve under in Egypt?",
     "عند من خدم يوسف عليه السلام في مصر؟",
     "In Egypt, Yusuf (AS) lived in the household of Al-Aziz, the minister, and served under him.",
     "في مصر عاش يوسف عليه السلام في بيت العزيز وزير مصر وخدم عنده.",
     [("Al-Aziz, the Minister of Egypt", "العزيز، وزير مصر", True),
      ("Pharaoh", "فرعون", False), ("His brothers", "إخوته", False),
      ("Prophet Ishaq (AS)", "النبي إسحاق عليه السلام", False)],
     ("Al-Aziz held which position?", "ما المنصب الذي شغله العزيز؟",
      [("Minister of Egypt", "وزير مصر", True), ("A merchant", "تاجر", False),
       ("A shepherd", "راعي غنم", False)])),
    ("q4", "identification", "r_alaziz_zulaikha", None,
     "Who is Zulaikha in this story?",
     "من هي زليخا في هذه القصة؟",
     "Zulaikha is the wife of Al-Aziz, part of the household Yusuf (AS) lived in.",
     "زليخا هي زوجة العزيز، من أهل البيت الذي عاش فيه يوسف عليه السلام.",
     [("The wife of Al-Aziz", "زوجة العزيز", True),
      ("The mother of Yusuf (AS)", "أم يوسف عليه السلام", False),
      ("The sister of Yaqub (AS)", "أخت يعقوب عليه السلام", False),
      ("A traveller who found the well", "مسافرة وجدت البئر", False)],
     None),
    ("q5", "comprehension", None, 2,
     "Why did Yusuf's brothers throw him into the well?",
     "لماذا ألقى إخوة يوسف في البئر؟",
     "The retrieved paragraph says the brothers decided to get rid of him and threw him into an abandoned well, after their father's love had stirred their jealousy.",
     "تذكر الفقرة المسترجعة أن الإخوة عزموا على التخلص منه فألقوه في بئر مهجورة، بعد أن أثار حبّ أبيهم له الحسد في نفوسهم.",
     [("They envied their father's love for him", "حسدوه على حب أبيه له", True),
      ("He refused to travel with them", "رفض السفر معهم", False),
      ("Al-Aziz asked them to", "طلب منهم العزيز ذلك", False),
      ("They were lost in the desert", "تاهوا في الصحراء", False)],
     ("What stirred the brothers' jealousy?", "ما الذي أثار حسد الإخوة؟",
      [("Their father's love for Yusuf", "حب أبيهم ليوسف", True),
       ("Yusuf's position in Egypt", "مكانة يوسف في مصر", False),
       ("A gift from Al-Aziz", "هدية من العزيز", False)])),
    ("q6", "comprehension", None, 1,
     "What did Yaqub (AS) warn Yusuf (AS) about?",
     "مِمَّ حذّر يعقوب عليه السلام يوسف عليه السلام؟",
     "The paragraph states that Yaqub warned Yusuf not to tell his brothers about the dream, fearing it would deepen their envy.",
     "تذكر الفقرة أن يعقوب حذّر يوسف من أن يخبر إخوته بالرؤيا خشية أن يزداد حسدهم.",
     [("Telling his brothers about his dream", "أن يخبر إخوته برؤياه", True),
      ("Travelling to Egypt", "السفر إلى مصر", False),
      ("Going near the well", "الاقتراب من البئر", False),
      ("Speaking to Al-Aziz", "التحدث إلى العزيز", False)],
     None),
]

PENDING_TEXT = (
    "This paragraph is waiting for text from a curated source.",
    "هذه الفقرة بانتظار نص من مصدر معتمد.",
)

def _paragraph_id(story_id: str, position: int) -> str:
    return f"{story_id}-p{position}"

def _entity_id(key: str) -> str:
    return f"e-{key}"

PLACEHOLDER_SOURCE_ID = "src-ui-placeholder"

def seed(session: Session) -> None:
    session.merge(
        Source(
            id=PLACEHOLDER_SOURCE_ID,
            title="UI design placeholder — not a curated source",
            citation="Text taken from the UI design mock-ups for development only.",
        )
    )

    for position, (story_id, level, characters, minutes, unlock, en_t, en_s, ar_t, ar_s) in enumerate(
        STORIES
    ):
        session.merge(
            Story(
                id=story_id,
                level=level,
                character_count=characters,
                minutes=minutes,
                unlock_after=unlock,
                position=position,
                published=True,
            )
        )
        en_heading, ar_heading = HEADINGS.get(story_id, ("", ""))
        session.merge(
            StoryTranslation(
                story_id=story_id, lang="en", title=en_t, summary=en_s,
                heading=en_heading, status="approved",
            )
        )
        session.merge(
            StoryTranslation(
                story_id=story_id, lang="ar", title=ar_t, summary=ar_s,
                heading=ar_heading, status="approved",
            )
        )

    for pid, symmetric, en_label, en_phrase, ar_label, ar_phrase in PREDICATES:
        session.merge(Predicate(id=pid, symmetric=symmetric))
        session.merge(
            PredicateTranslation(predicate_id=pid, lang="en", label=en_label, phrase=en_phrase)
        )
        session.merge(
            PredicateTranslation(predicate_id=pid, lang="ar", label=ar_label, phrase=ar_phrase)
        )

    for position, (en_body, ar_body, mentions) in enumerate(YUSUF_PARAGRAPHS):
        pid = _paragraph_id("yusuf", position)
        pending = en_body is None
        session.merge(
            Paragraph(id=pid, story_id="yusuf", position=position, is_pending=pending)
        )
        status = "draft" if pending else "approved"
        source = None if pending else PLACEHOLDER_SOURCE_ID
        session.merge(
            ParagraphTranslation(
                paragraph_id=pid,
                lang="en",
                body=en_body or PENDING_TEXT[0],
                status=status,
                source_id=source,
                approved_by="seed" if not pending else None,
            )
        )
        session.merge(
            ParagraphTranslation(
                paragraph_id=pid,
                lang="ar",
                body=ar_body or PENDING_TEXT[1],
                status=status,
                source_id=source,
                approved_by="seed" if not pending else None,
            )
        )
        for key in mentions:
            session.merge(
                EntityMention(
                    id=f"{pid}-{key}",
                    entity_id=_entity_id(key),
                    paragraph_id=pid,
                    lang="en",
                )
            )

    for key, entity_type, x, y, en, ar in YUSUF_ENTITIES:
        eid = _entity_id(key)
        session.merge(Entity(id=eid, key=key, type=entity_type, status="approved"))
        session.merge(
            StoryEntity(
                story_id="yusuf", entity_id=eid, layout_x=float(x), layout_y=float(y)
            )
        )
        session.merge(
            EntityTranslation(
                entity_id=eid, lang="en", label=en[0], category=en[1], description=en[2]
            )
        )
        session.merge(
            EntityTranslation(
                entity_id=eid, lang="ar", label=ar[0], category=ar[1], description=ar[2]
            )
        )

    session.flush()

    for rid, subject, predicate, obj in YUSUF_RELATIONS:
        session.merge(
            Relation(
                id=rid,
                subject_id=_entity_id(subject),
                predicate_id=predicate,
                object_id=_entity_id(obj),
                status="approved",
            )
        )
        session.merge(StoryRelation(story_id="yusuf", relation_id=rid))

    for position, (qid, qtype, relation, evidence, en_prompt, ar_prompt,
                   en_expl, ar_expl, options, followup) in enumerate(YUSUF_QUESTIONS):
        session.merge(
            Question(
                id=qid,
                story_id="yusuf",
                type=qtype,
                position=position,
                relation_id=relation,
                evidence_paragraph_id=(
                    _paragraph_id("yusuf", evidence) if evidence is not None else None
                ),
                status="approved",
            )
        )
        session.merge(
            QuestionTranslation(
                question_id=qid, lang="en", prompt=en_prompt, explanation=en_expl
            )
        )
        session.merge(
            QuestionTranslation(
                question_id=qid, lang="ar", prompt=ar_prompt, explanation=ar_expl
            )
        )
        for index, (en_label, ar_label, correct) in enumerate(options):
            oid = f"{qid}-o{index}"
            session.merge(
                Option(id=oid, question_id=qid, position=index, is_correct=correct)
            )
            session.merge(OptionTranslation(option_id=oid, lang="en", label=en_label))
            session.merge(OptionTranslation(option_id=oid, lang="ar", label=ar_label))

        if followup is None:
            continue

        fen, far, foptions = followup
        fid = f"{qid}f"
        session.merge(
            Question(
                id=fid,
                story_id="yusuf",
                type=qtype,
                position=position,
                relation_id=relation,
                followup_of=qid,
                status="approved",
            )
        )
        session.merge(QuestionTranslation(question_id=fid, lang="en", prompt=fen))
        session.merge(QuestionTranslation(question_id=fid, lang="ar", prompt=far))
        for index, (en_label, ar_label, correct) in enumerate(foptions):
            oid = f"{fid}-o{index}"
            session.merge(
                Option(id=oid, question_id=fid, position=index, is_correct=correct)
            )
            session.merge(OptionTranslation(option_id=oid, lang="en", label=en_label))
            session.merge(OptionTranslation(option_id=oid, lang="ar", label=ar_label))

def main() -> None:
    with SessionLocal() as session:
        seed(session)
        for lang in ("en", "ar"):
            index_story_chunks(session, "yusuf", lang)
        session.commit()
    print(
        f"Seeded {len(STORIES)} stories, {len(YUSUF_ENTITIES)} entities, "
        f"{len(YUSUF_RELATIONS)} relations, {len(YUSUF_QUESTIONS)} questions."
    )

if __name__ == "__main__":
    main()
