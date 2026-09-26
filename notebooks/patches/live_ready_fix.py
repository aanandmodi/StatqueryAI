# Run once in the EXISTING Kaggle kernel. No model reload, token change or tunnel restart.
from typing import Any
from fastapi.routing import APIRoute

assert "app" in globals(), "Run the model service cell first."
ready_routes = [(index, route) for index, route in enumerate(app.routes)
                if getattr(route, "path", None) == "/ready"]
assert len(ready_routes) == 1, "Expected exactly one readiness route."
ready_index, old_ready_route = ready_routes[0]
app.router.routes[ready_index] = APIRoute(
    "/ready", old_ready_route.dependant.call, methods=["GET"],
    response_model=dict[str, Any], name="ready"
)
app.openapi_schema = None
print("PASS: readiness schema repaired. Models and the existing tunnel are unchanged.")
