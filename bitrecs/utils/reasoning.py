import bittensor as bt
import requests
from dataclasses import dataclass, field    
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_dt

@dataclass
class ReasonReport:
    created_at: str = field(default_factory=str)
    miner_hotkey: str = field(default_factory=str)
    r_score: float = field(default=0.0)
    rank: int = field(default=0)

    
    @staticmethod
    def get_reports() -> list["ReasonReport"]:
        """
        Load latest offchain reasoning scores - public bucket
        """
        reports = []
        try:
            report_url = "https://pub-d5347166f7584bd88644018f6be5301f.r2.dev/r2_miner_reasons_report_testnet.json"
            report_json = requests.get(report_url).json()
            data = report_json.get("data", [])
            if not data or len(data) == 0:
                bt.logging.error("No data found in reasoning report")
                return []
            
            for item in data:
                report = ReasonReport(
                    created_at=item.get("created_at", ""),
                    miner_hotkey=item.get("miner_hotkey", ""),
                    r_score=item.get("r_score", 0.0),
                    rank=item.get("rank", 0)
                )
                reports.append(report)
            
            # old_reports = [r for r in reports if parse_dt(r.created_at) < datetime.now(timezone.utc) - timedelta(hours=12)]
            # if old_reports:
            #     bt.logging.error(f"Found {len(old_reports)} old reports - reason adjustment skipped")
            #     return []
                        
            sorted_reports = sorted(reports, key=lambda x: x.rank, reverse=False)
            return sorted_reports
        except Exception as e:
            bt.logging.error(f"load_user_actions Exception: {e}")     
        