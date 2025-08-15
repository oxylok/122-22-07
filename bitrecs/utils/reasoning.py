import os
import requests
import bittensor as bt
from dataclasses import dataclass, field    
from bitrecs import __version__ as this_version


@dataclass
class ReasonReport:
    created_at: str = field(default_factory=str)
    miner_hotkey: str = field(default_factory=str)
    f_score: float = field(default=0.0)
    evaluated: int = field(default=0)
    rank: int = field(default=0)

    
    @staticmethod
    def get_reports() -> list["ReasonReport"]:
        """
        Load latest reasoning scores
        """
        reports = []
        try:            
            proxy_url = os.environ.get("BITRECS_PROXY_URL").removesuffix("/")
            reason_url = f"{proxy_url}/reasoning"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('BITRECS_API_KEY')}",            
                'User-Agent': f'Bitrecs-Node/{this_version}'
            }        
            report_json = requests.get(reason_url, headers=headers).json()
            data = report_json.get("data", [])
            if not data or len(data) == 0:
                bt.logging.error("No data found in reasoning report")
                return []
            
            for item in data:
                report = ReasonReport(
                    created_at=item.get("created_at", ""),
                    miner_hotkey=item.get("miner_hotkey", ""),
                    evaluated=item.get("evaluated", 0),
                    f_score=item.get("f_score", 0.0),
                    rank=item.get("rank", 0)
                )
                reports.append(report)
            sorted_reports = sorted(reports, key=lambda x: x.rank, reverse=False)
            return sorted_reports
        except Exception as e:
            bt.logging.error(f"load_user_actions Exception: {e}")     
        