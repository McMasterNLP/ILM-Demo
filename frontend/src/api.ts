import { http } from "./lib/http";

export type Localized = { en: string; ar: string };

export type Level = "beginner" | "intermediate" | "advanced";

export type QuizResult = { correct: number; total: number };

export type StoryProgress = {
  paragraph: number;
  completed: boolean;
  quiz: QuizResult | null;
  quiz_in_progress: boolean;
};

export type Story = {
  id: string;
  level: Level;
  minutes: number;
  character_count: number;
  unlock_after: number | null;
  paragraph_count: number;
  question_count: number;
  title: Localized;
  summary: Localized;
  progress: StoryProgress;
};

export type Paragraph = {
  position: number;
  pending: boolean;
  text: Localized;
  characters: string[];
};

export type StoryDetail = Story & {
  heading: Localized;
  paragraphs: Paragraph[];
};

export type EntityType = "person" | "place";

export type GraphNode = {
  id: string;
  type: EntityType;
  label: Localized;
  category: Localized;
  description: Localized;
  x: number;
  y: number;
};

export type GraphEdge = {
  from: string;
  to: string;
  label: Localized;
  phrase: Localized;
  layer: "automatic" | "quran" | "legacy";
};

export type StoryGraphData = {
  story_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type QuestionType = "relationship" | "identification" | "comprehension";

export type Option = { id: string; label: Localized };

export type AnswerMode = "choice" | "open";

export type Question = {
  id: string;
  type: QuestionType;

  answer_mode: AnswerMode;
  position: number;
  prompt: Localized;
  options: Option[];
  evidence_paragraph: number | null;
};

export type Quiz = { story_id: string; questions: Question[] };

export type Attempt = {
  id: string;
  story_id: string;
  score: number;
  total: number;
  finished: boolean;
  answered_question_ids: string[];
};

export type AnswerResult = {
  is_correct: boolean;
  grading_status:
    | "correct"
    | "incorrect"
    | "not_enough_evidence"
    | "grading_unavailable";
  correct_option_id: string;

  feedback: string;
  reference_answer: string;
  explanation: Localized;
  kg_edge: string | null;
  source_text: string | null;
  follow_up: Question | null;
  score: number;
  answered: number;
  total: number;
};

export type ProgressEntry = {
  story_id: string;
  paragraph: number;
  completed: boolean;
  quiz: QuizResult | null;
};

export type LearnerSummary = {
  user_id: string;
  stories_started: number;
  stories_completed: number;
  questions_answered: number;
  progress: ProgressEntry[];
};

export const api = {
  listStories: () => http.get<Story[]>("/stories"),
  getStory: (storyId: string) => http.get<StoryDetail>(`/stories/${storyId}`),
  getGraph: (storyId: string) => http.get<StoryGraphData>(`/stories/${storyId}/graph`),

  getQuiz: (storyId: string, lang: string) =>
    http.get<Quiz>(`/quiz/${storyId}?lang=${lang}`),
  startAttempt: (
    storyId: string,
    lang: string,
    questionIds: string[]
  ) =>
    http.post<Attempt>(
      `/quiz/${storyId}/attempts?lang=${lang}`,
      { question_ids: questionIds }
    ),
  answer: (
    attemptId: string,
    body: {
      question_id: string;
      option_id?: string;
      text?: string;
      lang?: string;
      was_followup?: boolean;
    }
  ) => http.post<AnswerResult>(`/attempts/${attemptId}/answers`, body),
  finishAttempt: (attemptId: string) => http.post<Attempt>(`/attempts/${attemptId}/finish`),

  getProgress: () => http.get<LearnerSummary>("/me/progress"),
  setParagraph: (storyId: string, paragraph: number) =>
    http.put<ProgressEntry>(`/me/progress/${storyId}`, { paragraph }),
  restartStory: (storyId: string) => http.del<ProgressEntry>(`/me/progress/${storyId}`),
};

export function completedCount(stories: Story[]): number {
  return stories.filter((story) => story.progress.completed).length;
}

export function isLocked(story: Story, completed: number): boolean {
  return story.unlock_after !== null && completed < story.unlock_after;
}

export function percentOf(
  progress: Pick<StoryProgress, "paragraph" | "completed">,
  total: number
): number {
  if (total <= 0) return 0;
  if (progress.completed) return 100;
  if (progress.paragraph <= 0) return 0;
  return Math.round(((progress.paragraph + 1) / total) * 100);
}
