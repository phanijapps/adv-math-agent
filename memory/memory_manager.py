"""
Memory Management System for Math Agent
"""
import json
import logging
import sqlite3
import os
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class MemoryProvider:
    """Abstract base class for memory providers."""
    
    def store_memory(self, user_id: str, content: str, metadata: Optional[Dict] = None) -> str:
        """Store a memory and return its ID."""
        raise NotImplementedError
    
    def search_memories(self, user_id: str, query: str, limit: int = 10) -> List[Dict]:
        """Search memories for a user."""
        raise NotImplementedError
    
    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """Get a specific memory by ID."""
        raise NotImplementedError
    
    def update_memory(self, memory_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Update an existing memory."""
        raise NotImplementedError
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        raise NotImplementedError


class LocalMemoryProvider(MemoryProvider):
    """Local SQLite-based memory provider."""
    
    def __init__(self, db_path: str = "./data/memory.db"):
        self.db_path = db_path
        self._ensure_db_directory()
        self._initialize_db()
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _initialize_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create memories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    category TEXT DEFAULT 'general',
                    tags TEXT
                )
            ''')
            
            # Create search index
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_category 
                ON memories (user_id, category)
            ''')
            
            # Create problem solutions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS problem_solutions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    problem_hash TEXT NOT NULL,
                    problem_text TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    method TEXT,
                    tools_used TEXT,
                    confidence_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_count INTEGER DEFAULT 1,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create learning patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    difficulty_level TEXT,
                    success_rate REAL DEFAULT 0.0,
                    common_errors TEXT,
                    preferred_methods TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info(f"Initialized memory database at {self.db_path}")
    
    def store_memory(self, user_id: str, content: str, metadata: Optional[Dict] = None) -> str:
        """Store a memory."""
        memory_id = self._generate_memory_id(content)
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO memories 
                (id, user_id, content, metadata, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (memory_id, user_id, content, metadata_json))
            conn.commit()
        
        logger.info(f"Stored memory {memory_id} for user {user_id}")
        return memory_id
    
    def search_memories(self, user_id: str, query: str, limit: int = 10) -> List[Dict]:
        """Search memories using simple text matching."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, content, metadata, created_at, category
                FROM memories 
                WHERE user_id = ? AND (
                    content LIKE ? OR 
                    category LIKE ? OR
                    tags LIKE ?
                )
                ORDER BY updated_at DESC
                LIMIT ?
            ''', (user_id, f'%{query}%', f'%{query}%', f'%{query}%', limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'content': row[1],
                    'metadata': json.loads(row[2]) if row[2] else {},
                    'created_at': row[3],
                    'category': row[4],
                    'relevance': self._calculate_relevance(query, row[1])
                })
            
            # Sort by relevance
            results.sort(key=lambda x: x['relevance'], reverse=True)
            return results
    
    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """Get a specific memory."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, user_id, content, metadata, created_at, updated_at, category
                FROM memories WHERE id = ?
            ''', (memory_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'user_id': row[1],
                    'content': row[2],
                    'metadata': json.loads(row[3]) if row[3] else {},
                    'created_at': row[4],
                    'updated_at': row[5],
                    'category': row[6]
                }
            return None
    
    def update_memory(self, memory_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Update a memory."""
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE memories 
                SET content = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (content, metadata_json, memory_id))
            conn.commit()
            
            return cursor.rowcount > 0
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
            conn.commit()
            
            return cursor.rowcount > 0
    
    def store_problem_solution(self, user_id: str, problem: str, solution: str, 
                             method: str, tools_used: List[str], confidence: float = 1.0) -> str:
        """Store a problem-solution pair."""
        problem_hash = self._hash_problem(problem)
        solution_id = f"sol_{problem_hash}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if solution already exists
            cursor.execute('''
                SELECT id, accessed_count FROM problem_solutions 
                WHERE problem_hash = ? AND user_id = ?
            ''', (problem_hash, user_id))
            
            existing = cursor.fetchone()
            if existing:
                # Update access count and last accessed
                cursor.execute('''
                    UPDATE problem_solutions 
                    SET accessed_count = accessed_count + 1,
                        last_accessed = CURRENT_TIMESTAMP,
                        confidence_score = MAX(confidence_score, ?)
                    WHERE id = ?
                ''', (confidence, existing[0]))
            else:
                # Insert new solution
                cursor.execute('''
                    INSERT INTO problem_solutions 
                    (id, user_id, problem_hash, problem_text, solution, method, 
                     tools_used, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (solution_id, user_id, problem_hash, problem, solution, 
                      method, json.dumps(tools_used), confidence))
            
            conn.commit()
        
        return solution_id
    
    def find_similar_problems(self, user_id: str, problem: str, limit: int = 5) -> List[Dict]:
        """Find similar problems the user has solved before."""
        problem_hash = self._hash_problem(problem)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT problem_text, solution, method, tools_used, confidence_score,
                       accessed_count, created_at
                FROM problem_solutions 
                WHERE user_id = ? AND problem_hash != ?
                ORDER BY confidence_score DESC, accessed_count DESC
                LIMIT ?
            ''', (user_id, problem_hash, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'problem': row[0],
                    'solution': row[1],
                    'method': row[2],
                    'tools_used': json.loads(row[3]) if row[3] else [],
                    'confidence': row[4],
                    'access_count': row[5],
                    'created_at': row[6],
                    'similarity': self._calculate_problem_similarity(problem, row[0])
                })
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results
    
    def update_learning_pattern(self, user_id: str, topic: str, success: bool, 
                              method: str, error: Optional[str] = None):
        """Update learning patterns for a user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get existing pattern
            cursor.execute('''
                SELECT success_rate, common_errors, preferred_methods 
                FROM learning_patterns 
                WHERE user_id = ? AND topic = ?
            ''', (user_id, topic))
            
            existing = cursor.fetchone()
            if existing:
                # Update existing pattern
                current_rate = existing[0]
                new_rate = (current_rate + (1.0 if success else 0.0)) / 2.0
                
                errors = json.loads(existing[1]) if existing[1] else []
                if error and not success:
                    errors.append(error)
                
                methods = json.loads(existing[2]) if existing[2] else {}
                if method:
                    methods[method] = methods.get(method, 0) + 1
                
                cursor.execute('''
                    UPDATE learning_patterns 
                    SET success_rate = ?, common_errors = ?, preferred_methods = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND topic = ?
                ''', (new_rate, json.dumps(errors), json.dumps(methods), user_id, topic))
            else:
                # Create new pattern
                success_rate = 1.0 if success else 0.0
                errors = [error] if error and not success else []
                methods = {method: 1} if method else {}
                
                cursor.execute('''
                    INSERT INTO learning_patterns 
                    (id, user_id, topic, success_rate, common_errors, preferred_methods)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (f"pattern_{user_id}_{topic}", user_id, topic, success_rate,
                      json.dumps(errors), json.dumps(methods)))
            
            conn.commit()
    
    def get_learning_insights(self, user_id: str) -> Dict:
        """Get learning insights for a user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get learning patterns
            cursor.execute('''
                SELECT topic, success_rate, common_errors, preferred_methods
                FROM learning_patterns 
                WHERE user_id = ?
                ORDER BY last_updated DESC
            ''', (user_id,))
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'topic': row[0],
                    'success_rate': row[1],
                    'common_errors': json.loads(row[2]) if row[2] else [],
                    'preferred_methods': json.loads(row[3]) if row[3] else {}
                })
            
            # Get problem solving stats
            cursor.execute('''
                SELECT COUNT(*) as total_problems,
                       AVG(confidence_score) as avg_confidence,
                       COUNT(DISTINCT method) as methods_used
                FROM problem_solutions 
                WHERE user_id = ?
            ''', (user_id,))
            
            stats = cursor.fetchone()
            
            return {
                'learning_patterns': patterns,
                'total_problems_solved': stats[0] if stats else 0,
                'average_confidence': stats[1] if stats else 0.0,
                'methods_used': stats[2] if stats else 0
            }
    
    def _generate_memory_id(self, content: str) -> str:
        """Generate a unique memory ID."""
        return hashlib.md5(f"{content}_{datetime.now().isoformat()}".encode()).hexdigest()
    
    def _hash_problem(self, problem: str) -> str:
        """Generate a hash for a problem to detect similar problems."""
        # Normalize problem text for better matching
        normalized = problem.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)
    
    def _calculate_problem_similarity(self, problem1: str, problem2: str) -> float:
        """Calculate similarity between two problems."""
        words1 = set(problem1.lower().split())
        words2 = set(problem2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class MemoryManager:
    """Main memory manager that coordinates different providers."""
    
    def __init__(self, provider: Optional[MemoryProvider] = None):
        if provider is None:
            # Use local provider by default
            db_path = os.getenv("MEMORY_DATABASE_PATH", "./data/memory.db")
            provider = LocalMemoryProvider(db_path)
        
        self.provider = provider
        logger.info(f"Initialized memory manager with {type(provider).__name__}")
    
    def remember(self, user_id: str, content: str, category: str = "general", 
                tags: Optional[List[str]] = None) -> str:
        """Store a memory with categorization."""
        metadata = {
            "category": category,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat()
        }
        
        return self.provider.store_memory(user_id, content, metadata)
    
    def recall(self, user_id: str, query: str, limit: int = 10) -> List[Dict]:
        """Search and recall memories."""
        return self.provider.search_memories(user_id, query, limit)
    
    def remember_solution(self, user_id: str, problem: str, solution: str, 
                         method: str, tools_used: List[str], success: bool = True,
                         confidence: float = 1.0) -> str:
        """Remember a problem solution."""
        if hasattr(self.provider, 'store_problem_solution'):
            solution_id = self.provider.store_problem_solution(
                user_id, problem, solution, method, tools_used, confidence
            )
            
            # Update learning patterns
            if hasattr(self.provider, 'update_learning_pattern'):
                # Extract topic from problem (simple heuristic)
                topic = self._extract_topic(problem)
                self.provider.update_learning_pattern(user_id, topic, success, method)
            
            return solution_id
        else:
            # Fallback to regular memory storage
            content = f"Problem: {problem}\nSolution: {solution}\nMethod: {method}"
            return self.remember(user_id, content, "problem_solution")
    
    def find_similar_solutions(self, user_id: str, problem: str) -> List[Dict]:
        """Find similar problems and their solutions."""
        if hasattr(self.provider, 'find_similar_problems'):
            return self.provider.find_similar_problems(user_id, problem)
        else:
            # Fallback to memory search
            return self.recall(user_id, problem, 5)
    
    def get_learning_insights(self, user_id: str) -> Dict:
        """Get learning insights for the user."""
        if hasattr(self.provider, 'get_learning_insights'):
            return self.provider.get_learning_insights(user_id)
        else:
            return {"message": "Learning insights not available with current provider"}
    
    def _extract_topic(self, problem: str) -> str:
        """Extract mathematical topic from problem text."""
        problem_lower = problem.lower()
        
        # Simple keyword-based topic extraction
        if any(word in problem_lower for word in ['derivative', 'differentiate', 'diff']):
            return 'calculus_derivatives'
        elif any(word in problem_lower for word in ['integral', 'integrate', 'antiderivative']):
            return 'calculus_integrals'
        elif any(word in problem_lower for word in ['limit', 'approach', 'tends to']):
            return 'calculus_limits'
        elif any(word in problem_lower for word in ['solve', 'equation', 'root']):
            return 'algebra_equations'
        elif any(word in problem_lower for word in ['factor', 'factorize']):
            return 'algebra_factoring'
        elif any(word in problem_lower for word in ['probability', 'chance', 'odds']):
            return 'statistics_probability'
        elif any(word in problem_lower for word in ['mean', 'average', 'median', 'std']):
            return 'statistics_descriptive'
        elif any(word in problem_lower for word in ['triangle', 'circle', 'area', 'perimeter']):
            return 'geometry'
        elif any(word in problem_lower for word in ['matrix', 'vector', 'linear']):
            return 'linear_algebra'
        else:
            return 'general_math'
