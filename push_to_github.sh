#!/bin/bash

# Push to GitHub Script
# This script will commit and push your new Colab notebook to GitHub

echo "================================================"
echo "  Pushing LunarLander Colab Notebook to GitHub"
echo "================================================"
echo ""

# Add the new files
echo "📝 Adding new files..."
git add LunarLander_V2_Colab.ipynb
git add COLAB_UPLOAD_INSTRUCTIONS.md
git add PROFESSOR_QUICK_REFERENCE.md
git add SUBMISSION_GUIDE.md

# Check status
echo ""
echo "📊 Current git status:"
git status

# Commit
echo ""
echo "💾 Creating commit..."
git commit -m "Add Google Colab notebook for coursework submission

- Created comprehensive Colab notebook with all implementations
- Added upload instructions and professor reference guide
- Included submission guide with email template
- All code, results, and visualizations ready for review"

# Push to GitHub
echo ""
echo "🚀 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Successfully pushed to GitHub!"
echo ""
echo "Next steps:"
echo "1. Go to https://colab.research.google.com/"
echo "2. Upload LunarLander_V2_Colab.ipynb"
echo "3. Share the link with your professor"
echo ""
echo "See SUBMISSION_GUIDE.md for detailed instructions."
echo ""
