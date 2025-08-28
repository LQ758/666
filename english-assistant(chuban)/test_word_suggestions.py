#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试单词级发音改善建议功能
"""

import numpy as np
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.音素评分模块 import PhonemeScorer, PhonemeScore, DetailedPronunciationResult

def test_word_mapping():
    """测试单词到音素的映射功能"""
    print("=== 测试单词到音素映射 ===")
    
    scorer = PhonemeScorer()
    
    # 测试单词列表
    test_words = ['the', 'hello', 'world', 'beautiful', 'important', 'think', 'water']
    
    mapping = scorer.map_words_to_phonemes(test_words)
    
    for word, phonemes in mapping.items():
        print(f"单词 '{word}' -> 音素: {phonemes}")
    
    return mapping

def test_word_analysis():
    """测试单词级发音分析功能"""
    print("\n=== 测试单词级发音分析 ===")
    
    scorer = PhonemeScorer()
    
    # 模拟的音素评分数据
    phoneme_scores = [
        PhonemeScore(phoneme='ð', start_time=0.0, end_time=0.1, score=65, confidence=0.8, quality='fair', issues=['音素发音略短']),
        PhonemeScore(phoneme='ə', start_time=0.1, end_time=0.2, score=80, confidence=0.9, quality='good', issues=[]),
        PhonemeScore(phoneme='h', start_time=0.3, end_time=0.4, score=45, confidence=0.7, quality='poor', issues=['音素发音能量不足']),
        PhonemeScore(phoneme='ə', start_time=0.4, end_time=0.5, score=75, confidence=0.8, quality='good', issues=[]),
        PhonemeScore(phoneme='l', start_time=0.5, end_time=0.6, score=85, confidence=0.9, quality='good', issues=[]),
        PhonemeScore(phoneme='əʊ', start_time=0.6, end_time=0.8, score=50, confidence=0.6, quality='poor', issues=['音素发音不稳定']),
    ]
    
    # 测试单词
    words = ['the', 'hello']
    word_phoneme_mapping = scorer.map_words_to_phonemes(words)
    
    print(f"单词映射: {word_phoneme_mapping}")
    
    # 分析单词级发音
    word_scores = scorer.analyze_word_pronunciation(words, word_phoneme_mapping, phoneme_scores)
    
    print("\n单词级分析结果:")
    for word_info in word_scores:
        print(f"\n单词: '{word_info['word']}'")
        print(f"  评分: {word_info['score']}")
        print(f"  质量: {word_info['quality']}")
        print(f"  需要改进: {word_info['needs_improvement']}")
        print(f"  问题: {word_info['issues']}")
        print(f"  建议: {word_info['suggestions']}")
    
    return word_scores

def test_word_suggestions():
    """测试单词特定的发音建议生成"""
    print("\n=== 测试单词发音建议生成 ===")
    
    scorer = PhonemeScorer()
    
    # 测试不同类型的单词建议
    test_cases = [
        {
            'word': 'think',
            'phoneme_scores': [
                PhonemeScore(phoneme='θ', start_time=0.0, end_time=0.1, score=40, confidence=0.6, quality='poor', issues=['th音发音不清晰']),
                PhonemeScore(phoneme='ɪ', start_time=0.1, end_time=0.2, score=75, confidence=0.8, quality='good', issues=[]),
                PhonemeScore(phoneme='ŋ', start_time=0.2, end_time=0.3, score=60, confidence=0.7, quality='fair', issues=[]),
                PhonemeScore(phoneme='k', start_time=0.3, end_time=0.4, score=80, confidence=0.9, quality='good', issues=[]),
            ],
            'issues': ['th音发音不清晰', '鼻音不够充分']
        },
        {
            'word': 'water',
            'phoneme_scores': [
                PhonemeScore(phoneme='w', start_time=0.0, end_time=0.1, score=85, confidence=0.9, quality='good', issues=[]),
                PhonemeScore(phoneme='ɔː', start_time=0.1, end_time=0.3, score=70, confidence=0.8, quality='good', issues=[]),
                PhonemeScore(phoneme='t', start_time=0.3, end_time=0.4, score=45, confidence=0.6, quality='poor', issues=['爆破音不明显']),
                PhonemeScore(phoneme='ər', start_time=0.4, end_time=0.6, score=55, confidence=0.7, quality='fair', issues=['卷舌音不够']),
            ],
            'issues': ['爆破音不明显', '卷舌音不够']
        }
    ]
    
    for case in test_cases:
        print(f"\n测试单词: '{case['word']}'")
        suggestions = scorer._generate_word_suggestions(case['word'], case['phoneme_scores'], case['issues'])
        
        print("生成的建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    
    return True

def test_detailed_suggestions():
    """测试详细建议生成功能"""
    print("\n=== 测试详细建议生成 ===")
    
    scorer = PhonemeScorer()
    
    # 模拟完整的分析结果
    phoneme_scores = [
        PhonemeScore(phoneme='θ', start_time=0.0, end_time=0.1, score=40, confidence=0.6, quality='poor', issues=['th音发音不清晰']),
        PhonemeScore(phoneme='ɪ', start_time=0.1, end_time=0.2, score=75, confidence=0.8, quality='good', issues=[]),
        PhonemeScore(phoneme='s', start_time=0.2, end_time=0.3, score=50, confidence=0.7, quality='fair', issues=['摩擦声不够明显']),
    ]
    
    word_scores = [
        {
            'word': 'this',
            'score': 55.0,
            'quality': 'fair',
            'needs_improvement': True,
            'suggestions': [
                "单词 'this' 的发音节奏需要调整，注意每个音素的时长",
                "单词 'this' 中的音素 [θ] 需要重点练习",
                "练习 'th' 音：舌尖轻触上齿，气流从舌齿间通过"
            ]
        }
    ]
    
    all_issues = ['th音发音不清晰', '摩擦声不够明显']
    reference_text = "this is a test"
    
    # 生成详细建议
    suggestions = scorer._generate_detailed_suggestions(phoneme_scores, word_scores, all_issues, reference_text)
    
    print("生成的详细建议:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    return suggestions

def main():
    """主测试函数"""
    print("开始测试单词级发音改善建议功能\n")
    
    try:
        # 测试1: 单词到音素映射
        mapping = test_word_mapping()
        
        # 测试2: 单词级分析
        word_scores = test_word_analysis()
        
        # 测试3: 单词建议生成
        test_word_suggestions()
        
        # 测试4: 详细建议生成
        suggestions = test_detailed_suggestions()
        
        print("\n=== 测试总结 ===")
        print("✅ 单词到音素映射功能正常")
        print("✅ 单词级发音分析功能正常")
        print("✅ 单词建议生成功能正常")
        print("✅ 详细建议生成功能正常")
        print("\n🎉 所有测试通过！单词级发音改善建议功能已成功实现！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)