"""Generic, deterministic mock tools for demos and tests. No network access."""

from __future__ import annotations

import ast
import operator
from typing import Annotated
import httpx

from pydantic import Field

from agent_harness.errors import ToolError
from agent_harness.tools import ToolAnnotations, tool

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENT = 100


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ToolError(f"exponent larger than {_MAX_EXPONENT} is not allowed")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


@tool(annotations=ToolAnnotations(read_only=True, idempotent=True), concurrency_safe=True)
def calculator(expression: Annotated[str, Field(description="Arithmetic expression, e.g. '(2 + 3) * 4'")]) -> dict:
    """Evaluate an arithmetic expression using + - * / // % ** and parentheses."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc.msg}") from exc
    try:
        value = _evaluate(tree)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    return {"expression": expression, "result": value}


_WEATHER = {
    "london": {"condition": "cloudy", "temperature_c": 14},
    "paris": {"condition": "sunny", "temperature_c": 19},
    "tokyo": {"condition": "rainy", "temperature_c": 22},
    "bengaluru": {"condition": "partly cloudy", "temperature_c": 26},
}


@tool(annotations=ToolAnnotations(read_only=True, idempotent=True), concurrency_safe=True)
def get_weather(city: Annotated[str, Field(description="City name")]) -> dict:
    """Get the current weather for a city (mock data)."""
    report = _WEATHER.get(city.strip().lower())
    if report is None:
        raise ToolError(f"no weather data for '{city}'. Known cities: {', '.join(sorted(_WEATHER))}")
    return {"city": city, **report}


@tool(annotations=ToolAnnotations(read_only=True, idempotent=True), concurrency_safe=True)
async def web_search(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=10, description="Number of results")] = 3,
) -> list[dict]:
    """Search the web (mock results)."""
    return [
        {"title": f"Result {i} for '{query}'", "url": f"https://example.com/{i}", "snippet": f"Mock snippet {i}."}
        for i in range(1, max_results + 1)
    ]

@tool()
def get_trip_recommendations(
    location: str,
    start_date: str = None,
    end_date: str = None,
    num_people: int = 1,
    preferences: str = None,
    budget_per_person: float = None
) -> str:
    """
    Get comprehensive travel recommendations for a specific location using real-time data.
    
    Args:
        location: City name (e.g., "Udaipur", "Paris", "Tokyo")
        start_date: Trip start date in YYYY-MM-DD format (optional)
        end_date: Trip end date in YYYY-MM-DD format (optional)
        num_people: Number of travelers (default: 1)
        preferences: User preferences like "adventure, food, culture, relaxation, shopping, nature" (optional)
        budget_per_person: Budget per person in local currency (optional)
    
    Returns:
        Comprehensive trip plan with weather data, activities, and recommendations
    """
    import json
    from datetime import datetime, timedelta
    
    # If preferences not provided, use default
    if not preferences:
        preferences = "sightseeing, culture, food"
    
    # If dates not provided, use upcoming weekend
    if not start_date:
        today = datetime.now()
        days_until_friday = (4 - today.weekday()) % 7
        start_date = (today + timedelta(days=days_until_friday)).strftime("%Y-%m-%d")
    
    if not end_date:
        end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d")
    
    result = {
        "location": location,
        "trip_details": {
            "start_date": start_date,
            "end_date": end_date,
            "duration_days": (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days + 1,
            "num_people": num_people,
            "preferences": preferences.split(",") if isinstance(preferences, str) else preferences,
            "budget_per_person": budget_per_person
        },
        "weather_forecast": None,
        "recommendations": [],
        "tips": []
    }
    
    try:
        # Get weather data from OpenWeatherMap
        # Note: User needs to set OPENWEATHER_API_KEY environment variable
        # Get free API key from: https://openweathermap.org/api
        import os
        api_key = os.getenv("OPENWEATHER_API_KEY")
        
        if api_key:
            with httpx.Client(timeout=10.0) as client:
                # Use the forecast API directly with city name - it will find coordinates automatically
                weather_url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}&units=metric"
                
                weather_response = client.get(weather_url)
                
                if weather_response.status_code == 200:
                    weather_data = weather_response.json()
                    
                    # Extract city info (includes coordinates)
                    city_info = weather_data.get("city", {})
                    lat = city_info.get("coord", {}).get("lat")
                    lon = city_info.get("coord", {}).get("lon")
                    country = city_info.get("country")
                    city_name = city_info.get("name", location)
                    
                    # Parse weather forecasts
                    forecasts = []
                    for item in weather_data.get("list", [])[:8]:  # Next 24 hours
                        forecasts.append({
                            "datetime": item.get("dt_txt"),
                            "temp": f"{item['main']['temp']}°C",
                            "feels_like": f"{item['main']['feels_like']}°C",
                            "weather": item['weather'][0]['description'],
                            "humidity": f"{item['main']['humidity']}%",
                            "wind_speed": f"{item['wind']['speed']} m/s"
                        })
                    
                    result["weather_forecast"] = {
                        "location": f"{city_name}, {country}" if country else city_name,
                        "coordinates": {"lat": lat, "lon": lon},
                        "forecasts": forecasts,
                        "summary": f"Weather in {city_name}: {forecasts[0]['weather']}, {forecasts[0]['temp']}"
                    }
                elif weather_response.status_code == 404:
                    result["weather_forecast"] = {
                        "error": f"Could not find location: {location}",
                        "suggestion": "Try adding country name (e.g., 'Udaipur, India' or 'Paris, France')",
                        "status_code": 404
                    }
                elif weather_response.status_code == 401:
                    result["weather_forecast"] = {
                        "error": "Invalid API key",
                        "message": "Please check your OPENWEATHER_API_KEY in .env file",
                        "status_code": 401
                    }
                else:
                    result["weather_forecast"] = {
                        "error": f"Weather API error: {weather_response.status_code}",
                        "message": "Could not reach weather service"
                    }
        else:
            result["weather_forecast"] = {
                "note": "Weather data unavailable. Set OPENWEATHER_API_KEY environment variable.",
                "instructions": "Get free API key from https://openweathermap.org/api"
            }
    
    except Exception as e:
        result["weather_forecast"] = {"error": f"Failed to fetch weather: {str(e)}"}
    
    # Generate recommendations based on preferences
    pref_list = [p.strip().lower() for p in preferences.split(",")] if isinstance(preferences, str) else []
    
    recommendations = []
    
    # General recommendations
    recommendations.append({
        "category": "Getting Started",
        "suggestions": [
            f"Research local customs and etiquette for {location}",
            "Book accommodations in advance for better rates",
            "Check visa requirements and travel insurance",
            "Download offline maps and translation apps"
        ]
    })
    
    # Preference-based recommendations
    if "adventure" in pref_list or "nature" in pref_list:
        recommendations.append({
            "category": "Adventure & Nature",
            "suggestions": [
                "Look for hiking trails and outdoor activities",
                "Book adventure sports in advance (if available)",
                "Check weather conditions for outdoor activities",
                "Bring appropriate gear (hiking shoes, water bottles)"
            ]
        })
    
    if "food" in pref_list or "culture" in pref_list:
        recommendations.append({
            "category": "Food & Culture",
            "suggestions": [
                "Try local street food and traditional restaurants",
                "Join food walking tours to discover hidden gems",
                "Visit local markets and food halls",
                "Check Tripadvisor or Google reviews for top-rated restaurants"
            ]
        })
    
    if "sightseeing" in pref_list or "culture" in pref_list:
        recommendations.append({
            "category": "Sightseeing & Culture",
            "suggestions": [
                "Book skip-the-line tickets for popular attractions",
                "Consider guided tours for historical context",
                "Visit museums on free entry days (if available)",
                "Plan visits to main attractions early morning to avoid crowds"
            ]
        })
    
    if "relaxation" in pref_list:
        recommendations.append({
            "category": "Relaxation",
            "suggestions": [
                "Look for spa services and wellness centers",
                "Find peaceful parks or waterfront areas",
                "Book accommodations with relaxation amenities",
                "Schedule downtime between activities"
            ]
        })
    
    if "shopping" in pref_list:
        recommendations.append({
            "category": "Shopping",
            "suggestions": [
                "Visit local markets for authentic souvenirs",
                "Check opening hours of shopping districts",
                "Research tax refund policies for tourists",
                "Bring reusable bags for shopping"
            ]
        })
    
    result["recommendations"] = recommendations
    
    # Budget tips
    if budget_per_person:
        result["tips"].append(f"With a budget of {budget_per_person} per person, prioritize free attractions and local eateries")
    
    # Group travel tips
    if num_people > 1:
        result["tips"].extend([
            f"Traveling with {num_people} people - consider group discounts for activities",
            "Book group accommodations like vacation rentals for better value",
            "Use group transportation (minibus/van) for cost efficiency"
        ])
    
    # Weather-based tips
    if result.get("weather_forecast") and "forecasts" in result["weather_forecast"]:
        first_forecast = result["weather_forecast"]["forecasts"][0]
        if "rain" in first_forecast["weather"].lower():
            result["tips"].append("Pack rain gear and plan indoor activities")
        
        temp = float(first_forecast["temp"].replace("°C", ""))
        if temp > 30:
            result["tips"].append("Hot weather expected - stay hydrated and use sun protection")
        elif temp < 10:
            result["tips"].append("Cold weather expected - pack warm clothing")
    
    # External resources
    result["useful_resources"] = {
        "weather": f"https://openweathermap.org/city/{location.replace(' ', '-')}",
        "tripadvisor": f"https://www.tripadvisor.com/Search?q={location.replace(' ', '+')}",
        "google_maps": f"https://www.google.com/maps/search/?api=1&query={location.replace(' ', '+')}",
        "booking": f"https://www.booking.com/searchresults.html?ss={location.replace(' ', '+')}",
        "wikitravel": f"https://wikitravel.org/en/{location.replace(' ', '_')}"
    }
    
    return json.dumps(result, indent=2, default=str)


@tool()
def get_country_info(country_name: str) -> dict:
    """
    Fetch country information (currency, languages, etc.) from the public RestCountries API.
    Useful for travel planning.
    """
    # Using a public API for demonstration
    url = f"https://restcountries.com/v3.1/name/{country_name}"
    try:
        with httpx.Client() as client:
            response = client.get(url)
            if response.status_code == 404:
                return {"error": "Country not found"}
            response.raise_for_status()
            data = response.json()[0]
            return {
                "name": data.get("name", {}).get("common"),
                "capital": data.get("capital", ["N/A"])[0],
                "region": data.get("region"),
                "currencies": list(data.get("currencies", {}).keys()),
                "languages": list(data.get("languages", {}).values())
            }
    except Exception as e:
        return {"error": f"Failed to fetch country info: {str(e)}"}