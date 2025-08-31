// Statistics Management Module
class StatsManager {
    constructor() {
        this.episodeHistory = [];
        this.elements = {};
    }

    setElements(elements) {
        this.elements = elements;
    }

    resetStats() {
        this.episodeHistory = [];
        this.updateEpisodeResults();
    }

    addEpisode(score, lines) {
        this.episodeHistory.push({ score: score, lines: lines });
        this.updateEpisodeResults();
        console.log('Episode added:', { score, lines, total: this.episodeHistory.length });
    }

    updateEpisodeResults() {
        const totalEpisodes = this.episodeHistory.length;
        
        if (totalEpisodes === 0) {
            this.safeUpdateElement('totalEpisodes', '0');
            this.safeUpdateElement('avgScore', '0.0');
            this.safeUpdateElement('totalLines', '0');
            this.safeUpdateElement('bestScore', '0');
            return;
        }
        
        const totalScore = this.episodeHistory.reduce((sum, ep) => sum + ep.score, 0);
        const totalLines = this.episodeHistory.reduce((sum, ep) => sum + ep.lines, 0);
        const avgScore = totalScore / totalEpisodes;
        const bestScore = Math.max(...this.episodeHistory.map(ep => ep.score));
        
        this.safeUpdateElement('totalEpisodes', totalEpisodes.toString());
        this.safeUpdateElement('avgScore', avgScore.toFixed(1));
        this.safeUpdateElement('totalLines', totalLines.toString());
        this.safeUpdateElement('bestScore', bestScore.toString());
    }

    safeUpdateElement(id, value) {
        const element = this.elements[id];
        if (element) {
            element.textContent = value;
        } else {
            console.warn(`Element not found: ${id}`);
        }
    }

    updateCurrentStats(score, lines, steps, status) {
        this.safeUpdateElement('currentScore', score.toString());
        this.safeUpdateElement('currentLines', lines.toString());
        this.safeUpdateElement('currentSteps', steps.toString());
        this.safeUpdateElement('gameStatus', status);
    }
}
