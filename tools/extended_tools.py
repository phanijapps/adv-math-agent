"""
Additional mathematical tools for specialized computations
Pure Python implementation using advanced mathematical libraries
"""
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Union, Tuple
import os

logger = logging.getLogger(__name__)

try:
    from langchain.tools import BaseTool
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for when dependencies aren't installed
    BaseTool = object
    BaseModel = object
    Field = lambda **kwargs: None

# Advanced math libraries with fallbacks
try:
    import sympy as sp
    import numpy as np
    from scipy import stats, optimize, integrate, special
    ADVANCED_MATH_AVAILABLE = True
except ImportError:
    ADVANCED_MATH_AVAILABLE = False
    logger.warning("Advanced math libraries not available. Some features will be limited.")


class AdvancedMathToolInput(BaseModel):
    query: str = Field(description="Mathematical query in natural language")
    operation_type: Optional[str] = Field(default=None, description="Type of operation: calculus, algebra, analysis, physics")


class AdvancedMathTool(BaseTool):
    """Advanced mathematical computations using pure Python libraries."""
    
    name = "advanced_math_tool"
    description = """
    Performs advanced mathematical computations using SymPy and SciPy including:
    - Advanced calculus and analysis
    - Complex analysis and functions
    - Differential equations
    - Linear algebra operations
    - Special functions
    - Mathematical physics
    - Symbolic and numerical computation
    
    Input: query (natural language mathematical question), operation_type (optional)
    """
    args_schema = AdvancedMathToolInput
    
    def _run(self, query: str, operation_type: Optional[str] = None) -> str:
        if not ADVANCED_MATH_AVAILABLE:
            return json.dumps({
                "error": "Advanced math libraries not available",
                "suggestion": "Install sympy, numpy, and scipy for advanced computations",
                "basic_result": self._basic_fallback(query)
            })
        
        try:
            # Parse and classify the query
            query_type = operation_type or self._classify_query(query)
            
            # Route to appropriate handler
            if "calculus" in query_type.lower():
                result = self._handle_calculus(query)
            elif "algebra" in query_type.lower() or "equation" in query.lower():
                result = self._handle_algebra(query)
            elif "analysis" in query_type.lower() or "limit" in query.lower():
                result = self._handle_analysis(query)
            elif "physics" in query_type.lower():
                result = self._handle_physics(query)
            elif "complex" in query.lower():
                result = self._handle_complex_analysis(query)
            elif "differential" in query.lower():
                result = self._handle_differential_equations(query)
            elif "special" in query.lower() or "function" in query.lower():
                result = self._handle_special_functions(query)
            else:
                result = self._handle_general_math(query)
            
            return json.dumps({
                "query": query,
                "query_type": query_type,
                "result": result,
                "source": "SymPy/SciPy",
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Advanced math tool error: {e}")
            return json.dumps({
                "error": f"Advanced math computation failed: {str(e)}",
                "query": query,
                "fallback_result": self._basic_fallback(query)
            })
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of mathematical query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['derivative', 'integral', 'differentiate', 'integrate']):
            return "calculus"
        elif any(word in query_lower for word in ['solve', 'equation', 'system', 'root']):
            return "algebra"
        elif any(word in query_lower for word in ['limit', 'series', 'convergence', 'asymptotic']):
            return "analysis"
        elif any(word in query_lower for word in ['force', 'energy', 'wave', 'field', 'physics']):
            return "physics"
        elif any(word in query_lower for word in ['complex', 'imaginary', 'real part']):
            return "complex"
        elif any(word in query_lower for word in ['differential equation', 'ode', 'pde']):
            return "differential"
        elif any(word in query_lower for word in ['gamma', 'beta', 'bessel', 'special function']):
            return "special"
        else:
            return "general"
    
    def _handle_calculus(self, query: str) -> Dict[str, Any]:
        """Handle calculus-related queries."""
        try:
            # Extract mathematical expressions
            expressions = self._extract_expressions(query)
            if not expressions:
                return {"error": "No mathematical expression found in query"}
            
            expr_str = expressions[0]
            x = sp.Symbol('x')
            expr = sp.sympify(expr_str)
            
            results = {}
            
            if 'derivative' in query.lower() or 'differentiate' in query.lower():
                # Handle derivatives
                order = self._extract_derivative_order(query)
                derivative = sp.diff(expr, x, order)
                results['derivative'] = {
                    'expression': str(expr),
                    'order': order,
                    'result': str(derivative),
                    'latex': sp.latex(derivative),
                    'simplified': str(sp.simplify(derivative))
                }
                
                # Add numerical evaluation at key points
                try:
                    eval_points = [0, 1, -1]
                    evaluations = {}
                    for point in eval_points:
                        val = float(derivative.subs(x, point))
                        evaluations[f'f\'({point})'] = val
                    results['derivative']['evaluations'] = evaluations
                except:
                    pass
            
            if 'integral' in query.lower() or 'integrate' in query.lower():
                # Handle integrals
                indefinite = sp.integrate(expr, x)
                results['integral'] = {
                    'expression': str(expr),
                    'indefinite': str(indefinite),
                    'latex': sp.latex(indefinite)
                }
                
                # Try definite integral if bounds are mentioned
                bounds = self._extract_bounds(query)
                if bounds:
                    try:
                        definite = sp.integrate(expr, (x, bounds[0], bounds[1]))
                        results['integral']['definite'] = {
                            'bounds': bounds,
                            'result': str(definite),
                            'numerical': float(definite) if definite.is_real else None
                        }
                    except:
                        pass
            
            return results
            
        except Exception as e:
            return {"error": f"Calculus computation failed: {str(e)}"}
    
    def _handle_algebra(self, query: str) -> Dict[str, Any]:
        """Handle algebra-related queries."""
        try:
            results = {}
            
            if 'solve' in query.lower():
                equations = self._extract_equations(query)
                if equations:
                    eq_str = equations[0]
                    
                    # Determine variables
                    variables = self._extract_variables(eq_str)
                    if not variables:
                        variables = ['x']
                    
                    var_symbols = [sp.Symbol(var) for var in variables]
                    
                    try:
                        eq = sp.sympify(eq_str)
                        solutions = sp.solve(eq, var_symbols[0] if len(var_symbols) == 1 else var_symbols)
                        
                        results['equation_solving'] = {
                            'equation': eq_str,
                            'variables': variables,
                            'solutions': [str(sol) for sol in solutions] if isinstance(solutions, list) else str(solutions),
                            'numerical_solutions': self._get_numerical_solutions(solutions)
                        }
                        
                        # Add verification
                        if isinstance(solutions, list) and solutions:
                            verification = {}
                            for i, sol in enumerate(solutions[:3]):  # Verify first 3 solutions
                                try:
                                    substituted = eq.subs(var_symbols[0], sol)
                                    verification[f'solution_{i+1}'] = {
                                        'value': str(sol),
                                        'verification': str(sp.simplify(substituted)),
                                        'is_valid': sp.simplify(substituted) == 0
                                    }
                                except:
                                    pass
                            results['equation_solving']['verification'] = verification
                        
                    except Exception as e:
                        results['equation_solving'] = {"error": f"Could not solve equation: {str(e)}"}
            
            return results
            
        except Exception as e:
            return {"error": f"Algebra computation failed: {str(e)}"}
    
    def _handle_analysis(self, query: str) -> Dict[str, Any]:
        """Handle mathematical analysis queries."""
        try:
            results = {}
            
            if 'limit' in query.lower():
                expressions = self._extract_expressions(query)
                if expressions:
                    expr_str = expressions[0]
                    x = sp.Symbol('x')
                    expr = sp.sympify(expr_str)
                    
                    # Extract limit point
                    limit_point = self._extract_limit_point(query)
                    if limit_point is None:
                        limit_point = 0
                    
                    # Calculate limits
                    try:
                        limit_val = sp.limit(expr, x, limit_point)
                        left_limit = sp.limit(expr, x, limit_point, '-')
                        right_limit = sp.limit(expr, x, limit_point, '+')
                        
                        results['limit_analysis'] = {
                            'expression': str(expr),
                            'point': limit_point,
                            'limit': str(limit_val),
                            'left_limit': str(left_limit),
                            'right_limit': str(right_limit),
                            'exists': left_limit == right_limit == limit_val,
                            'numerical_value': float(limit_val) if limit_val.is_real else None
                        }
                    except Exception as e:
                        results['limit_analysis'] = {"error": f"Limit computation failed: {str(e)}"}
            
            if 'series' in query.lower():
                expressions = self._extract_expressions(query)
                if expressions:
                    expr_str = expressions[0]
                    x = sp.Symbol('x')
                    expr = sp.sympify(expr_str)
                    
                    # Taylor series expansion
                    try:
                        center = 0  # Default center
                        order = 6   # Default order
                        
                        series = sp.series(expr, x, center, order)
                        
                        results['series_analysis'] = {
                            'expression': str(expr),
                            'center': center,
                            'order': order,
                            'series': str(series),
                            'latex': sp.latex(series),
                            'coefficients': self._extract_series_coefficients(series)
                        }
                    except Exception as e:
                        results['series_analysis'] = {"error": f"Series computation failed: {str(e)}"}
            
            return results
            
        except Exception as e:
            return {"error": f"Analysis computation failed: {str(e)}"}
    
    def _handle_physics(self, query: str) -> Dict[str, Any]:
        """Handle physics-related mathematical queries."""
        try:
            results = {}
            
            # Common physics constants
            constants = {
                'c': 299792458,  # speed of light
                'g': 9.81,       # gravitational acceleration
                'h': 6.62607015e-34,  # Planck constant
                'k': 1.380649e-23,    # Boltzmann constant
                'e': 1.602176634e-19, # elementary charge
            }
            
            # Physics equations patterns
            if 'kinetic energy' in query.lower():
                results['physics_formula'] = {
                    'concept': 'Kinetic Energy',
                    'formula': 'KE = (1/2) * m * v²',
                    'variables': {'m': 'mass (kg)', 'v': 'velocity (m/s)'},
                    'units': 'Joules (J)'
                }
            
            elif 'potential energy' in query.lower():
                results['physics_formula'] = {
                    'concept': 'Gravitational Potential Energy',
                    'formula': 'PE = m * g * h',
                    'variables': {'m': 'mass (kg)', 'g': 'acceleration due to gravity (m/s²)', 'h': 'height (m)'},
                    'units': 'Joules (J)'
                }
            
            elif 'wave' in query.lower():
                results['physics_formula'] = {
                    'concept': 'Wave Equation',
                    'formula': 'v = f * λ',
                    'variables': {'v': 'wave speed (m/s)', 'f': 'frequency (Hz)', 'λ': 'wavelength (m)'},
                    'alternative_forms': ['f = v/λ', 'λ = v/f']
                }
            
            # Add dimensional analysis if numbers are present
            numbers = re.findall(r'\d+(?:\.\d+)?', query)
            if numbers:
                results['dimensional_analysis'] = {
                    'extracted_values': numbers,
                    'suggestion': 'Check units for dimensional consistency'
                }
            
            return results
            
        except Exception as e:
            return {"error": f"Physics computation failed: {str(e)}"}
    
    def _handle_complex_analysis(self, query: str) -> Dict[str, Any]:
        """Handle complex analysis queries."""
        try:
            results = {}
            
            expressions = self._extract_expressions(query)
            if expressions:
                expr_str = expressions[0]
                z = sp.Symbol('z', complex=True)
                
                try:
                    expr = sp.sympify(expr_str.replace('i', 'I'))  # SymPy uses I for imaginary unit
                    
                    results['complex_analysis'] = {
                        'expression': str(expr),
                        'real_part': str(sp.re(expr)),
                        'imaginary_part': str(sp.im(expr)),
                        'magnitude': str(sp.Abs(expr)),
                        'argument': str(sp.arg(expr)),
                        'conjugate': str(sp.conjugate(expr))
                    }
                    
                    # Evaluate at specific points if possible
                    test_points = [1, sp.I, 1+sp.I, -1, -sp.I]
                    evaluations = {}
                    for point in test_points:
                        try:
                            val = expr.subs(z, point)
                            evaluations[str(point)] = {
                                'value': str(val),
                                'numerical': complex(val) if val.is_number else None
                            }
                        except:
                            pass
                    
                    if evaluations:
                        results['complex_analysis']['evaluations'] = evaluations
                    
                except Exception as e:
                    results['complex_analysis'] = {"error": f"Complex analysis failed: {str(e)}"}
            
            return results
            
        except Exception as e:
            return {"error": f"Complex analysis computation failed: {str(e)}"}
    
    def _handle_differential_equations(self, query: str) -> Dict[str, Any]:
        """Handle differential equations."""
        try:
            results = {}
            
            # This is a simplified DE handler - could be expanded significantly
            if 'differential equation' in query.lower() or 'ode' in query.lower():
                results['differential_equations'] = {
                    'note': 'Differential equation solving requires specific equation format',
                    'suggestion': 'Provide equation in form like "y\' + y = 0" or "d²y/dx² + y = sin(x)"',
                    'methods': ['separation of variables', 'integrating factor', 'characteristic equation', 'power series']
                }
                
                # Try to extract and solve simple ODEs
                if "y'" in query or "dy/dx" in query:
                    x = sp.Symbol('x')
                    y = sp.Function('y')
                    
                    # Very basic pattern matching - could be greatly expanded
                    if "y' + y = 0" in query:
                        eq = sp.Eq(y(x).diff(x) + y(x), 0)
                        solution = sp.dsolve(eq, y(x))
                        results['differential_equations']['example_solution'] = {
                            'equation': "y' + y = 0",
                            'solution': str(solution),
                            'general_form': 'y = C₁ * e^(-x)'
                        }
            
            return results
            
        except Exception as e:
            return {"error": f"Differential equation computation failed: {str(e)}"}
    
    def _handle_special_functions(self, query: str) -> Dict[str, Any]:
        """Handle special mathematical functions."""
        try:
            results = {}
            
            if 'gamma' in query.lower():
                # Extract number for gamma function
                numbers = re.findall(r'\d+(?:\.\d+)?', query)
                if numbers:
                    n = float(numbers[0])
                    gamma_val = special.gamma(n)
                    results['gamma_function'] = {
                        'input': n,
                        'gamma_value': float(gamma_val),
                        'property': f'Γ({n}) = {gamma_val:.6f}',
                        'note': 'Γ(n) = (n-1)! for positive integers'
                    }
            
            elif 'beta' in query.lower():
                numbers = re.findall(r'\d+(?:\.\d+)?', query)
                if len(numbers) >= 2:
                    a, b = float(numbers[0]), float(numbers[1])
                    beta_val = special.beta(a, b)
                    results['beta_function'] = {
                        'inputs': [a, b],
                        'beta_value': float(beta_val),
                        'property': f'B({a}, {b}) = {beta_val:.6f}',
                        'relation': 'B(a,b) = Γ(a)Γ(b)/Γ(a+b)'
                    }
            
            elif 'bessel' in query.lower():
                results['bessel_functions'] = {
                    'types': ['J_n(x) - Bessel function of first kind', 'Y_n(x) - Bessel function of second kind'],
                    'note': 'Specify order n and argument x for numerical evaluation',
                    'applications': ['cylindrical wave equations', 'heat conduction', 'vibrations']
                }
            
            return results
            
        except Exception as e:
            return {"error": f"Special function computation failed: {str(e)}"}
    
    def _handle_general_math(self, query: str) -> Dict[str, Any]:
        """Handle general mathematical queries."""
        try:
            results = {}
            
            # Extract and evaluate expressions
            expressions = self._extract_expressions(query)
            if expressions:
                for i, expr_str in enumerate(expressions):
                    try:
                        expr = sp.sympify(expr_str)
                        results[f'expression_{i+1}'] = {
                            'original': expr_str,
                            'parsed': str(expr),
                            'simplified': str(sp.simplify(expr)),
                            'expanded': str(sp.expand(expr)),
                            'factored': str(sp.factor(expr)) if expr.is_polynomial() else 'Not applicable'
                        }
                        
                        # Try numerical evaluation
                        try:
                            numerical = float(expr)
                            results[f'expression_{i+1}']['numerical_value'] = numerical
                        except:
                            pass
                            
                    except Exception as e:
                        results[f'expression_{i+1}'] = {"error": f"Could not process expression: {str(e)}"}
            
            return results
            
        except Exception as e:
            return {"error": f"General math computation failed: {str(e)}"}
    
    # Helper methods
    def _extract_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text."""
        # Enhanced pattern for mathematical expressions
        patterns = [
            r'[a-zA-Z0-9\+\-\*/\^\(\)\s]+(?:[=<>≤≥≠]+[a-zA-Z0-9\+\-\*/\^\(\)\s]+)*',
            r'[xy]\^?\d*[\+\-\*/][xy\d\+\-\*/\^\(\)]+',
            r'sin\([^)]+\)|cos\([^)]+\)|tan\([^)]+\)|ln\([^)]+\)|log\([^)]+\)',
            r'\d*[xy]\^?\d*',
            r'e\^\([^)]+\)'
        ]
        
        expressions = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            expressions.extend(matches)
        
        # Clean and filter expressions
        cleaned = []
        for expr in expressions:
            expr = expr.strip()
            if len(expr) > 2 and any(c in expr for c in 'xy+-*/^()'):
                cleaned.append(expr)
        
        return list(set(cleaned))  # Remove duplicates
    
    def _extract_equations(self, text: str) -> List[str]:
        """Extract equations from text."""
        eq_patterns = [
            r'[^=]*=[^=]*',
            r'[a-zA-Z0-9\+\-\*/\^\(\)\s]+=\s*[a-zA-Z0-9\+\-\*/\^\(\)\s]+',
        ]
        
        equations = []
        for pattern in eq_patterns:
            matches = re.findall(pattern, text)
            equations.extend(matches)
        
        return [eq.strip() for eq in equations if '=' in eq and len(eq.strip()) > 3]
    
    def _extract_variables(self, text: str) -> List[str]:
        """Extract variable names from mathematical text."""
        variables = re.findall(r'\b[a-zA-Z]\b', text)
        return list(set(var for var in variables if var not in ['e', 'i', 'pi']))
    
    def _extract_derivative_order(self, text: str) -> int:
        """Extract derivative order from text."""
        if 'second' in text.lower() or "d²" in text or "'''" in text:
            return 2
        elif 'third' in text.lower() or "d³" in text:
            return 3
        else:
            return 1
    
    def _extract_bounds(self, text: str) -> Optional[Tuple[float, float]]:
        """Extract integration bounds from text."""
        bound_patterns = [
            r'from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)',
            r'between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)',
            r'\[(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\]'
        ]
        
        for pattern in bound_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return (float(match.group(1)), float(match.group(2)))
        
        return None
    
    def _extract_limit_point(self, text: str) -> Optional[float]:
        """Extract limit point from text."""
        patterns = [
            r'as\s+x\s+approaches\s+(-?\d+(?:\.\d+)?)',
            r'x\s*→\s*(-?\d+(?:\.\d+)?)',
            r'limit.*?x.*?(-?\d+(?:\.\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return float(match.group(1))
        
        if 'infinity' in text.lower():
            return sp.oo
        
        return None
    
    def _get_numerical_solutions(self, solutions) -> List[float]:
        """Get numerical values of solutions."""
        numerical = []
        if isinstance(solutions, list):
            for sol in solutions:
                try:
                    val = float(sol)
                    numerical.append(val)
                except:
                    pass
        else:
            try:
                val = float(solutions)
                numerical.append(val)
            except:
                pass
        
        return numerical
    
    def _extract_series_coefficients(self, series) -> List[float]:
        """Extract coefficients from a series expansion."""
        try:
            coeffs = []
            # This is a simplified extraction - could be enhanced
            str_series = str(series)
            numbers = re.findall(r'-?\d+(?:\.\d+)?', str_series)
            return [float(n) for n in numbers[:6]]  # First 6 coefficients
        except:
            return []
    
    def _basic_fallback(self, query: str) -> Dict[str, Any]:
        """Basic fallback computation without advanced libraries."""
        try:
            # Extract numbers and basic operations
            numbers = re.findall(r'-?\d+(?:\.\d+)?', query)
            if len(numbers) >= 2:
                a, b = float(numbers[0]), float(numbers[1])
                
                basic_ops = {
                    'sum': a + b,
                    'difference': a - b,
                    'product': a * b,
                    'quotient': a / b if b != 0 else 'undefined',
                    'power': a ** b if abs(b) < 10 else 'result too large'
                }
                
                return {
                    'basic_arithmetic': basic_ops,
                    'note': 'Install sympy and scipy for advanced mathematical computations'
                }
            
            return {'note': 'Basic computation requires numerical values'}
            
        except Exception as e:
            return {'error': f'Basic computation failed: {str(e)}'}


class PhysicsToolInput(BaseModel):
    concept: str = Field(description="Physics concept or equation")
    values: Optional[Dict] = Field(default=None, description="Known values for calculation")


class PhysicsTool(BaseTool):
    """Physics calculations and formula tool."""
    
    name = "physics_tool"
    description = """
    Provides physics formulas and calculations including:
    - Classical mechanics (kinematics, dynamics, energy)
    - Thermodynamics
    - Electromagnetism
    - Waves and optics
    - Modern physics basics
    
    Input: concept, values (optional)
    """
    args_schema = PhysicsToolInput
    
    def _run(self, concept: str, values: Optional[Dict] = None) -> str:
        try:
            concept_lower = concept.lower()
            
            # Physics formulas database
            formulas = {
                'kinetic_energy': {
                    'formula': 'KE = (1/2) * m * v²',
                    'variables': {'m': 'mass (kg)', 'v': 'velocity (m/s)'},
                    'units': 'Joules (J)'
                },
                'potential_energy': {
                    'formula': 'PE = m * g * h',
                    'variables': {'m': 'mass (kg)', 'g': 'gravity (9.81 m/s²)', 'h': 'height (m)'},
                    'units': 'Joules (J)'
                },
                'force': {
                    'formula': 'F = m * a',
                    'variables': {'m': 'mass (kg)', 'a': 'acceleration (m/s²)'},
                    'units': 'Newtons (N)'
                },
                'wave_speed': {
                    'formula': 'v = f * λ',
                    'variables': {'v': 'wave speed (m/s)', 'f': 'frequency (Hz)', 'λ': 'wavelength (m)'},
                    'units': 'm/s'
                },
                'ohms_law': {
                    'formula': 'V = I * R',
                    'variables': {'V': 'voltage (V)', 'I': 'current (A)', 'R': 'resistance (Ω)'},
                    'units': 'Volts (V)'
                }
            }
            
            # Match concept to formula
            matched_formula = None
            for key, formula_data in formulas.items():
                if any(word in concept_lower for word in key.split('_')):
                    matched_formula = formula_data
                    break
            
            if not matched_formula:
                # Try partial matching
                for key, formula_data in formulas.items():
                    if key.replace('_', ' ') in concept_lower:
                        matched_formula = formula_data
                        break
            
            result = {
                'concept': concept,
                'formula_found': matched_formula is not None
            }
            
            if matched_formula:
                result.update(matched_formula)
                
                # Calculate if values provided
                if values:
                    try:
                        calculation = self._calculate_physics(key, values)
                        result['calculation'] = calculation
                    except Exception as e:
                        result['calculation_error'] = str(e)
            else:
                result['available_concepts'] = list(formulas.keys())
                result['suggestion'] = 'Try one of the available physics concepts'
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Physics tool error: {e}")
            return json.dumps({"error": f"Physics calculation failed: {str(e)}"})
    
    def _calculate_physics(self, formula_type: str, values: Dict) -> Dict:
        """Calculate physics values based on formula type."""
        if formula_type == 'kinetic_energy':
            if 'm' in values and 'v' in values:
                ke = 0.5 * values['m'] * values['v'] ** 2
                return {'kinetic_energy': ke, 'units': 'J'}
        
        elif formula_type == 'potential_energy':
            if 'm' in values and 'h' in values:
                g = values.get('g', 9.81)
                pe = values['m'] * g * values['h']
                return {'potential_energy': pe, 'units': 'J'}
        
        elif formula_type == 'force':
            if 'm' in values and 'a' in values:
                force = values['m'] * values['a']
                return {'force': force, 'units': 'N'}
        
        elif formula_type == 'wave_speed':
            if 'f' in values and 'λ' in values:
                speed = values['f'] * values['λ']
                return {'wave_speed': speed, 'units': 'm/s'}
        
        elif formula_type == 'ohms_law':
            if 'I' in values and 'R' in values:
                voltage = values['I'] * values['R']
                return {'voltage': voltage, 'units': 'V'}
        
        return {'error': 'Insufficient values for calculation'}


class GeometryToolInput(BaseModel):
    shape: str = Field(description="Geometric shape (circle, triangle, rectangle, etc.)")
    operation: str = Field(description="Operation (area, perimeter, volume, etc.)")
    parameters: Dict = Field(description="Shape parameters")


class GeometryTool(BaseTool):
    """Geometric calculations tool."""
    
    name = "geometry_tool"
    description = """
    Performs geometric calculations including:
    - Area and perimeter calculations
    - Volume and surface area
    - Coordinate geometry
    - Trigonometric calculations
    
    Input: shape, operation, parameters
    """
    args_schema = GeometryToolInput
    
    def _run(self, shape: str, operation: str, parameters: Dict) -> str:
        try:
            if shape.lower() == "circle":
                return self._circle_calculations(operation, parameters)
            elif shape.lower() == "triangle":
                return self._triangle_calculations(operation, parameters)
            elif shape.lower() == "rectangle":
                return self._rectangle_calculations(operation, parameters)
            elif shape.lower() == "sphere":
                return self._sphere_calculations(operation, parameters)
            else:
                return json.dumps({"error": f"Shape '{shape}' not supported"})
                
        except Exception as e:
            logger.error(f"Geometry tool error: {e}")
            return json.dumps({"error": f"Geometry calculation failed: {str(e)}"})
    
    def _circle_calculations(self, operation: str, params: Dict) -> str:
        """Circle calculations."""
        import math
        
        if operation.lower() == "area":
            radius = params.get("radius")
            if radius is not None:
                area = math.pi * radius ** 2
                return json.dumps({
                    "shape": "circle",
                    "operation": "area",
                    "radius": radius,
                    "result": area,
                    "formula": "π × r²"
                })
        
        elif operation.lower() == "circumference":
            radius = params.get("radius")
            if radius is not None:
                circumference = 2 * math.pi * radius
                return json.dumps({
                    "shape": "circle",
                    "operation": "circumference",
                    "radius": radius,
                    "result": circumference,
                    "formula": "2π × r"
                })
        
        return json.dumps({"error": "Invalid operation or missing parameters"})
    
    def _triangle_calculations(self, operation: str, params: Dict) -> str:
        """Triangle calculations."""
        import math
        
        if operation.lower() == "area":
            # Heron's formula or base × height
            if "base" in params and "height" in params:
                area = 0.5 * params["base"] * params["height"]
                return json.dumps({
                    "shape": "triangle",
                    "operation": "area",
                    "base": params["base"],
                    "height": params["height"],
                    "result": area,
                    "formula": "½ × base × height"
                })
            elif "a" in params and "b" in params and "c" in params:
                # Heron's formula
                a, b, c = params["a"], params["b"], params["c"]
                s = (a + b + c) / 2
                area = math.sqrt(s * (s - a) * (s - b) * (s - c))
                return json.dumps({
                    "shape": "triangle",
                    "operation": "area",
                    "sides": [a, b, c],
                    "result": area,
                    "formula": "Heron's formula"
                })
        
        return json.dumps({"error": "Invalid operation or missing parameters"})
    
    def _rectangle_calculations(self, operation: str, params: Dict) -> str:
        """Rectangle calculations."""
        if operation.lower() == "area":
            length = params.get("length")
            width = params.get("width")
            if length is not None and width is not None:
                area = length * width
                return json.dumps({
                    "shape": "rectangle",
                    "operation": "area",
                    "length": length,
                    "width": width,
                    "result": area,
                    "formula": "length × width"
                })
        
        elif operation.lower() == "perimeter":
            length = params.get("length")
            width = params.get("width")
            if length is not None and width is not None:
                perimeter = 2 * (length + width)
                return json.dumps({
                    "shape": "rectangle",
                    "operation": "perimeter",
                    "length": length,
                    "width": width,
                    "result": perimeter,
                    "formula": "2 × (length + width)"
                })
        
        return json.dumps({"error": "Invalid operation or missing parameters"})
    
    def _sphere_calculations(self, operation: str, params: Dict) -> str:
        """Sphere calculations."""
        import math
        
        radius = params.get("radius")
        if radius is None:
            return json.dumps({"error": "Radius parameter required"})
        
        if operation.lower() == "volume":
            volume = (4/3) * math.pi * radius ** 3
            return json.dumps({
                "shape": "sphere",
                "operation": "volume",
                "radius": radius,
                "result": volume,
                "formula": "⁴⁄₃ × π × r³"
            })
        
        elif operation.lower() == "surface_area":
            surface_area = 4 * math.pi * radius ** 2
            return json.dumps({
                "shape": "sphere",
                "operation": "surface_area",
                "radius": radius,
                "result": surface_area,
                "formula": "4π × r²"
            })
        
        return json.dumps({"error": "Invalid operation"})


class NumberTheoryToolInput(BaseModel):
    number: int = Field(description="Number to analyze")
    operation: str = Field(description="Operation (prime_check, factorization, gcd, lcm)")
    second_number: Optional[int] = Field(default=None, description="Second number for operations like GCD")


class NumberTheoryTool(BaseTool):
    """Number theory operations tool."""
    
    name = "number_theory_tool"
    description = """
    Performs number theory operations including:
    - Prime number checking
    - Prime factorization
    - GCD and LCM calculations
    - Modular arithmetic
    
    Input: number, operation, second_number (optional)
    """
    args_schema = NumberTheoryToolInput
    
    def _run(self, number: int, operation: str, second_number: Optional[int] = None) -> str:
        try:
            if operation.lower() == "prime_check":
                return self._prime_check(number)
            elif operation.lower() == "factorization":
                return self._prime_factorization(number)
            elif operation.lower() == "gcd" and second_number is not None:
                return self._gcd(number, second_number)
            elif operation.lower() == "lcm" and second_number is not None:
                return self._lcm(number, second_number)
            else:
                return json.dumps({"error": f"Invalid operation: {operation}"})
                
        except Exception as e:
            logger.error(f"Number theory tool error: {e}")
            return json.dumps({"error": f"Number theory calculation failed: {str(e)}"})
    
    def _prime_check(self, n: int) -> str:
        """Check if a number is prime."""
        if n < 2:
            is_prime = False
        elif n == 2:
            is_prime = True
        elif n % 2 == 0:
            is_prime = False
        else:
            is_prime = True
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    is_prime = False
                    break
        
        return json.dumps({
            "number": n,
            "operation": "prime_check",
            "is_prime": is_prime,
            "explanation": f"{n} is {'prime' if is_prime else 'composite'}"
        })
    
    def _prime_factorization(self, n: int) -> str:
        """Find prime factorization."""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        
        # Count factors
        factor_counts = {}
        for factor in factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        return json.dumps({
            "number": n if n > 1 else factors[0] if len(factors) == 1 else factors[0] * factors[1] if len(factors) == 2 else "original_number",
            "operation": "prime_factorization",
            "factors": factors,
            "factor_counts": factor_counts,
            "factorization": " × ".join([f"{p}^{e}" if e > 1 else str(p) for p, e in factor_counts.items()])
        })
    
    def _gcd(self, a: int, b: int) -> str:
        """Calculate greatest common divisor."""
        import math
        result = math.gcd(a, b)
        
        return json.dumps({
            "numbers": [a, b],
            "operation": "gcd",
            "result": result
        })
    
    def _lcm(self, a: int, b: int) -> str:
        """Calculate least common multiple."""
        import math
        result = abs(a * b) // math.gcd(a, b) if a != 0 and b != 0 else 0
        
        return json.dumps({
            "numbers": [a, b],
            "operation": "lcm",
            "result": result
        })


# Additional tool collection function
def get_extended_math_tools() -> List:
    """Get extended mathematical tools."""
    tools = []
    
    try:
        tools.extend([
            AdvancedMathTool(),
            GeometryTool(),
            NumberTheoryTool(),
            PhysicsTool()
        ])
    except Exception as e:
        logger.warning(f"Some extended tools could not be loaded: {e}")
    
    return tools
