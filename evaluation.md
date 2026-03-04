### 0. Commencer par SVM Tfid :
- 10k :
    - Dev matched accuracy: 0.4333
    - Dev mismatched accuracy: 0.4374
- 50k :
    - Dev matched accuracy: 0.4772
    - Dev mismatched accuracy: 0.4662

Tester KFold Cross Validation :
- 10k :
    - Dev matched accuracy: 0.4555
    - Dev mismatched accuracy: 0.4683
- 50k : Ca sert a rien car Kfold c'est juste pour le petit dataset

Le TF-IDF ne comprend pas l'ordre des mots ni le sens profond (sémantique), ce qui est indispensable pour le NLI.

Au lieu de juste concaténer [v1, v2], les chercheurs utilisent souvent la différence absolue |v1 - v2| ou le produit élément par élément v1 * v2 pour forcer le modèle à comparer les deux phrases.

### 1. Concaténation + différence absolue + produit élément par élément*
[v1 ; v2 ; |v1-v2| ; v1*v2] pour forcer le modèle à comparer les deux phrases
- 10k :
    - Dev matched accuracy: 0.4793
    - Dev mismatched accuracy: 0.4876
- 50k :
    - Dev matched accuracy: 0.5164
    - Dev mismatched accuracy: 0.5165

### 2. Passer aux Transformers
???