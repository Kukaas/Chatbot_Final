<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Carbon\Carbon;

class TroubleshootingGuidesSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $troubleshootingData = [
            [
                'issue' => 'Slow Performance',
                'solution' => 'Step 1: Open Task Manager (Ctrl + Shift + Esc on Windows, or Command + Space, then type "Activity Monitor" on macOS).
                               Step 2: Check the "Processes" tab to identify resource-intensive applications.
                               Step 3: End tasks for unnecessary processes by selecting them and clicking "End Task" (Windows) or "Quit" (macOS).',
            ],
            [
                'issue' => 'Hardware Failure',
                'solution' => 'Step 1: Check physical connections of cables and peripherals. Ensure power cables, display cables, and other peripherals are securely connected.
                               Step 2: Restart the computer to see if the issue persists.
                               Step 3: If the problem continues, run hardware diagnostics. Most computers have built-in diagnostics accessed by pressing a specific key during startup (e.g., F12 for Dell, F2 for HP).',
            ],
            [
                'issue' => 'Peripheral Connectivity',
                'solution' => 'Step 1: Unplug and replug the peripheral device.
                               Step 2: Try connecting the device to a different USB port or using a different cable.
                               Step 3: Check device drivers in Device Manager (Windows) or System Information (macOS) to ensure proper installation and functionality.',
                'created_at' => Carbon::now(),
                'updated_at' => Carbon::now(),
            ],
            [
                'issue' => 'Battery Drain',
                'solution' => 'Step 1: Check battery usage statistics in device settings to identify apps consuming excessive power.
                               Step 2: Close background apps by accessing the app switcher and swiping them away.
                               Step 3: Disable features like location services and push notifications for apps that are not essential.',
                'created_at' => Carbon::now(),
                'updated_at' => Carbon::now(),
            ],
            [
                'issue' => 'Screen Damage',
                'solution' => 'Step 1: Backup important data using cloud storage, computer backup software, or a data transfer cable.
                               Step 2: Contact the device manufacturer or a certified repair center for screen replacement options.
                               Step 3: If immediate replacement is not possible, consider using screen protectors or adhesive patches to prevent further damage.',
                'created_at' => Carbon::now(),
                'updated_at' => Carbon::now(),
            ],
            [
                'issue' => 'Connectivity Issues',
                'solution' => 'Step 1: Toggle Airplane Mode on/off to reset network connections.
                               Step 2: Restart the device and router.
                               Step 3: Reset network settings in device settings to clear any conflicting configurations.',
                'created_at' => Carbon::now(),
                'updated_at' => Carbon::now(),
            ],
            [
                'issue' => 'Overheating',
                'solution' => 'Step 1: Check server vents and fans for dust buildup. Use compressed air to clean if necessary.
                               Step 2: Ensure proper airflow by organizing cables and removing obstructions around server vents.
                               Step 3: Monitor server temperature using built-in management tools or third-party software.',
                'created_at' => Carbon::now(),
                'updated_at' => Carbon::now(),
            ],
            [
                'issue' => 'Disk Space Management',
                'solution' => 'Step 1: Sort files by size to identify large files taking up space.
                               Step 2: Delete temporary files, such as those in the Temp folder (Windows) or Trash (macOS).
                               Step 3: Uninstall unused applications to free up additional space.',
                'created_at' => Carbon::now(),
                'updated_at' => Carbon::now(),
            ],
            [
                'issue' => 'Slow Internet Speed',
                'solution' => 'Step 1: Check internet speed using online speed testing tools like Speedtest.net or Fast.com.
                               Step 2: Restart the modem and router to refresh network connections.
                               Step 3: Contact the ISP to report slow speeds and request assistance.',
                'created_at' => Carbon::now(),
                'updated_at' => Carbon::now(),
            ],
            [
                'issue' => 'Update Failures',
                'solution' => 'Step 1: Check internet connection and ensure it\'s stable.
                               Step 2: Clear space on the system drive if it\'s low on storage.
                               Step 3: Manually download and install updates from the official website if automatic updates fail.',
                'created_at' => Carbon::now(),
                'updated_at' => Carbon::now(),
            ],
        ];

        foreach ($troubleshootingData as $data) {
            try {
                // Generate embedding
                $embedding = $this->generateEmbedding($data['issue'] . ' ' . $data['solution']);

                DB::table('troubleshooting_guides')->insert([
                    'issue' => $data['issue'],
                    'solution' => $data['solution'],
                    'embedding' => $embedding,
                    'created_at' => Carbon::now(),
                    'updated_at' => Carbon::now(),
                ]);
            } catch (\Exception $e) {
                echo "Error processing record '{$data['issue']}': {$e->getMessage()}\n";
            }
        }
    }

    private function generateEmbedding($text)
    {
        // Call the FastAPI service to generate embeddings
        $response = Http::post('http://127.0.0.1:8000/generate-embedding', [
            'text' => $text
        ]);

        if ($response->successful()) {
            return json_encode($response->json()['embedding']);
        } else {
            throw new \Exception('Failed to generate embedding: ' . $response->body());
        }
    }
} 