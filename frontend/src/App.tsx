import React, { useState } from "react";
import { Sidebar } from "./components/common/Sidebar";
import { TopBar } from "./components/common/TopBar";
import { TabId } from "./types";
import { OverviewView } from "./views/OverviewView";
import { LiveSimulationView } from "./views/LiveSimulationView";
import { PolicyArenaView } from "./views/PolicyArenaView";
import { NetworkLabView } from "./views/NetworkLabView";
import { ExperimentsView } from "./views/ExperimentsView";
import { ModelLabView } from "./views/ModelLabView";
import { VideoPlayerView } from "./views/VideoPlayerView";
import { SettingsView } from "./views/SettingsView";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>("live-simulation");
  const [activeProfile, setActiveProfile] = useState<string>("Sudden Drop");

  const handleNavigateToSim = (profileId: string) => {
    setActiveProfile(profileId);
    setActiveTab("live-simulation");
  };

  return (
    <div className="flex h-screen bg-background overflow-hidden font-sans text-text-primary">
      {/* Persistent Left Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        simulationRunning={activeTab === "live-simulation"}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Compact Sticky Topbar */}
        <TopBar
          activeProfileName={activeProfile}
          onQuickRun={() => setActiveTab("policy-arena")}
        />

        {/* Scrollable Viewport */}
        <main className="flex-1 overflow-y-auto">
          {activeTab === "overview" && <OverviewView onNavigate={setActiveTab} />}
          {activeTab === "live-simulation" && <LiveSimulationView />}
          {activeTab === "policy-arena" && <PolicyArenaView />}
          {activeTab === "network-lab" && (
            <NetworkLabView onNavigateToSim={handleNavigateToSim} />
          )}
          {activeTab === "experiments" && <ExperimentsView />}
          {activeTab === "model-lab" && <ModelLabView />}
          {activeTab === "video-player" && <VideoPlayerView />}
          {activeTab === "settings" && <SettingsView />}
        </main>
      </div>
    </div>
  );
};

export default App;

