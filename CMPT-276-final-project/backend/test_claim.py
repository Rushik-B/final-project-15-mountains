#!/usr/bin/env python3
"""
Simple test script for the Factify API that simulates a user submitting a claim for verification.
This script helps test the API functionality after removing the social context feature.
"""

import argparse
import json
import requests
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

# Initialize Rich console for better output formatting
console = Console()

# Default API URL (assumes the Flask app is running locally)
DEFAULT_API_URL = "http://localhost:8080"

def verify_claim(api_url, claim):
    """Submit a claim to the verification API and return the results"""
    
    endpoint = f"{api_url}/api/verification/claim"
    
    console.print(f"\n[bold cyan]Submitting claim for verification:[/bold cyan] {claim}")
    
    try:
        start_time = time.time()
        response = requests.post(
            endpoint,
            json={"claim": claim},
            timeout=180  # Long timeout since verification can take time
        )
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', {})
            
            # Display a summary of the results
            console.print(Panel.fit(
                f"[bold green]Verification Complete[/bold green]\n"
                f"Processing time: {data.get('processing_time_seconds', 0):.2f} seconds\n"
                f"Request time: {request_time:.2f} seconds",
                title="Factify Verification",
                border_style="green"
            ))
            
            # Create a table for the verification results
            table = Table(title="Claim Verification Results")
            table.add_column("Property", style="cyan")
            table.add_column("Value")
            
            table.add_row("Original Claim", claim)
            table.add_row("Verdict", f"[bold]{result.get('verdict', 'Unknown')}[/bold]")
            table.add_row("Confidence", f"{result.get('overall_confidence', 0):.2f}")
            
            # Display sub-claims
            sub_claims = result.get('sub_claims', [])
            table.add_row("Sub-claims", f"{len(sub_claims)} sub-claims analyzed")
            
            console.print(table)
            
            # Show detailed information for each sub-claim
            if sub_claims:
                console.print("\n[bold]Sub-claim Analysis:[/bold]")
                
                for i, sub_claim in enumerate(sub_claims):
                    sc_table = Table(title=f"Sub-claim {i+1}")
                    sc_table.add_column("Property", style="cyan")
                    sc_table.add_column("Value")
                    
                    sc_table.add_row("Text", sub_claim.get('sub_claim', 'N/A'))
                    
                    eval_data = sub_claim.get('evaluation', {})
                    stance = eval_data.get('stance', 'N/A')
                    stance_color = {
                        'supports': 'green',
                        'refutes': 'red',
                        'insufficient': 'yellow',
                        'error': 'red'
                    }.get(stance, 'white')
                    
                    sc_table.add_row("Stance", f"[{stance_color}]{stance}[/{stance_color}]")
                    sc_table.add_row("Confidence", f"{eval_data.get('confidence_score', 0):.2f}")
                    
                    # Show evidence count
                    evidence_count = sub_claim.get('evidence_count', len(sub_claim.get('evidence', [])))
                    sc_table.add_row("Evidence Items", f"{evidence_count}")
                    
                    # Show key points (limited to first 3 for readability)
                    key_support = eval_data.get('key_support_points', [])
                    if key_support:
                        points = "\n".join([f"• {point}" for point in key_support[:3]])
                        if len(key_support) > 3:
                            points += f"\n• ... and {len(key_support) - 3} more"
                        sc_table.add_row("Key Support Points", points)
                    
                    key_refute = eval_data.get('key_refutation_points', [])
                    if key_refute:
                        points = "\n".join([f"• {point}" for point in key_refute[:3]])
                        if len(key_refute) > 3:
                            points += f"\n• ... and {len(key_refute) - 3} more"
                        sc_table.add_row("Key Refutation Points", points)
                    
                    console.print(sc_table)
                    
                    # Display all papers for this sub-claim
                    evidence_items = sub_claim.get('evidence', [])
                    if evidence_items:
                        console.print(f"\n[bold]Papers for Sub-claim {i+1} ({len(evidence_items)} papers):[/bold]")
                        
                        for j, paper in enumerate(evidence_items):
                            paper_panel = Panel(
                                f"[bold cyan]Title:[/bold cyan] {paper.get('title', 'No title')}\n\n"
                                f"[bold cyan]Authors:[/bold cyan] {', '.join(paper.get('authors', ['Unknown']))}\n\n"
                                f"[bold cyan]Publication Date:[/bold cyan] {paper.get('publication_date') or paper.get('year') or 'Unknown'}\n\n"
                                f"[bold cyan]URL:[/bold cyan] {paper.get('url') or 'Not available'}\n\n"
                                f"[bold cyan]Abstract:[/bold cyan]\n{paper.get('abstract', 'No abstract available')}",
                                title=f"Paper {j+1}",
                                border_style="blue"
                            )
                            console.print(paper_panel)
                    
                    console.print("")
            
            console.print("\n[italic]Would you like to view the full JSON response? (y/n)[/italic]", end=" ")
            choice = input().lower()
            if choice == 'y':
                console.print_json(json.dumps(data, indent=2))
            
            return True
            
        else:
            console.print(f"[bold red]Error:[/bold red] {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Connection error:[/bold red] {str(e)}")
        console.print("[yellow]Make sure the Flask app is running and accessible at the specified URL.[/yellow]")
        return False
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the Factify API with a claim verification request")
    parser.add_argument("--url", default=DEFAULT_API_URL, help=f"API base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--claim", default="Coffee consumption reduces the risk of diabetes", 
                        help="Claim to verify (default: 'Coffee consumption reduces the risk of diabetes')")
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold]Factify API Test[/bold]\n"
        f"API URL: {args.url}\n"
        f"Claim: {args.claim}",
        title="Factify Claim Verification Test",
        border_style="blue"
    ))
    
    # First check if the API is available
    try:
        health_response = requests.get(f"{args.url}/health", timeout=5)
        if health_response.status_code == 200:
            console.print("[green]✓ API is available[/green]")
        else:
            console.print("[red]✗ API returned non-200 status code[/red]")
            return False
    except requests.exceptions.RequestException:
        console.print("[red]✗ Could not connect to the API[/red]")
        console.print("[yellow]Make sure the Flask app is running and accessible at the specified URL.[/yellow]")
        return False
    
    # Then verify the claim
    result = verify_claim(args.url, args.claim)
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 