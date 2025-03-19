import React, { useState, useEffect } from 'react';
import ResearchPaperCard from './ResearchPaperCard'; // Import your ResearchPaperCard component

const PaperDisplay = () => {
  const [paperData, setPaperData] = useState(null);
  const [loading, setLoading] = useState(true);

  // Mock paper data (instead of fetching from an API)
  useEffect(() => {
    const mockData = {
      title: "The Possibility of Artificial Sweeteners Replacing Added Sugar in the Diet of High Sugar Consumption Patients with Cardiovascular Disease",
      author: "Xinyang Iong",
      date: "28/12/2023",
      categories: ["Research paper"],
      abstract: `Artificial sweeteners have emerged as popular alternatives to traditional sweeteners, driven by the growing concern over sugar consumption and its associated rise in obesity and metabolic disorders. Despite their widespread use, the safety and health implications of artificial sweeteners remain a topic of debate, with conflicting contributing to uncertainty about their long-term effects. This review synthesizes current scientific evidence regarding the impact of artificial sweeteners on gut microbiota and gastrointestinal health.

Our analysis of in vitro experiments, animal models, and clinical trials reveals that artificial sweeteners can alter the composition and abundance of gut microbes. These changes raise concerns about their potential to affect overall gut health and contribute to gastrointestinal disorders. Additionally, artificial sweeteners have been shown to influence the production of metabolites by gut bacteria, further impacting systemic health.

The findings suggest that artificial sweeteners may have complex and sometimes contradictory effects on gut microbiota. While some studies indicate potential benefits, such as reduced caloric intake and weight management, others highlight detrimental effects on microbial balance and metabolic functions. The inconsistent results underscore the need for further research to comprehensively understand the physiological impacts of various artificial sweeteners on human health. Future studies should aim for long-term, well-controlled investigations to clarify these relationships, ensuring evidence-based guidelines for the safe use of artificial sweeteners in diet management AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA`,
      publisher: "Society of Chemical Industry",
      badgeText: "Crossref"
    };

    setPaperData(mockData);
    setLoading(false);
  }, []);

  if (loading) return <div>Loading...</div>;
  if (!paperData) return <div>No paper found</div>;

  return <ResearchPaperCard {...paperData} />;
};

export default PaperDisplay;
