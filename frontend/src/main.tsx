import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import "./index.css";
import App from "./App";
import { LangProvider } from "./lib/i18n";
import Stories from "./pages/Stories";
import StoryRead from "./pages/StoryRead";
import StoryMap from "./pages/StoryMap";
import StoryMaps from "./pages/StoryMaps";
import Quiz from "./pages/Quiz";
import Quizzes from "./pages/Quizzes";
import Progress from "./pages/Progress";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <LangProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<App />}>
            <Route index element={<Stories />} />
            <Route path="story/:storyId" element={<StoryRead />} />
            <Route path="map" element={<StoryMaps />} />
            <Route path="story/:storyId/map" element={<StoryMap />} />
            <Route path="quiz" element={<Quizzes />} />
            <Route path="quiz/:storyId" element={<Quiz />} />
            <Route path="progress" element={<Progress />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </LangProvider>
  </React.StrictMode>
);
